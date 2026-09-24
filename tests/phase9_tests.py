import io, sys, json
from pathlib import Path
# Ensure project root is on sys.path so app can be imported when running tests
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app, load_risk_results
import pandas as pd

app = create_app()
client = app.test_client()

results = []

def record(name, expected, resp_status, resp_text_snippet, passed, notes=''):
    results.append({'test':name,'expected':expected,'status':resp_status,'snippet':resp_text_snippet[:200],'pass':passed,'notes':notes})

# GET /
rv = client.get('/')
record('GET /','200 and dashboard content', rv.status_code, rv.get_data(as_text=True), rv.status_code==200 and 'Automated GST Invoice Fraud Detection System' in rv.get_data(as_text=True))

# GET /results
rv = client.get('/results')
text = rv.get_data(as_text=True)
record('GET /results','200 and results table', rv.status_code, text, rv.status_code==200 and ('Invoice No' in text or 'Results' in text))

# GET /invoice/INV000001
rv = client.get('/invoice/INV000001')
text = rv.get_data(as_text=True)
record('GET /invoice/INV000001','200 and invoice details page', rv.status_code, text, rv.status_code==200 and 'Invoice Information' in text)

# GET /evaluation
rv = client.get('/evaluation')
text = rv.get_data(as_text=True)
record('GET /evaluation','200 and evaluation content', rv.status_code, text, rv.status_code==200 and ('Model comparison' in text or 'Method' in text))

# GET /upload
rv = client.get('/upload')
text = rv.get_data(as_text=True)
record('GET /upload','200 and upload form', rv.status_code, text, rv.status_code==200 and 'Upload' in text)

# Prepare files for POSTs
orig = pd.read_csv('data/gst_invoices.csv')
small = orig.head(5)
small_csv = small.to_csv(index=False)
missing = orig.iloc[:3,:5].to_csv(index=False)
wrong = 'col1,col2\n1,2\n'

# POST valid CSV upload
data = {'file':(io.BytesIO(small_csv.encode('utf-8')), 'test_valid.csv')}
rv = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
text = rv.get_data(as_text=True)
passed = rv.status_code in (200,302, 200)
record('POST /upload valid CSV','redirect to uploaded results or 200', rv.status_code, text, passed)

# POST missing columns
data = {'file':(io.BytesIO(missing.encode('utf-8')), 'test_missing.csv')}
rv = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
text = rv.get_data(as_text=True)
expected_fail = 'Missing required columns' in text or rv.status_code==400
record('POST /upload missing cols','400 or missing columns error', rv.status_code, text, expected_fail)

# POST wrong extension
data = {'file':(io.BytesIO(wrong.encode('utf-8')), 'test.txt')}
rv = client.post('/upload', data=data, content_type='multipart/form-data', follow_redirects=True)
text = rv.get_data(as_text=True)
expected_fail = 'Only CSV files are allowed' in text or rv.status_code==400
record('POST /upload wrong extension','400 and only csv error', rv.status_code, text, expected_fail)

# GET invalid invoice
rv = client.get('/invoice/INVALID999')
record('GET /invoice/INVALID999','404 page not found', rv.status_code, rv.get_data(as_text=True), rv.status_code==404)

# Static files
rv = client.get('/static/css/style.css')
record('GET static css','200', rv.status_code, rv.get_data(as_text=True), rv.status_code==200)
rv = client.get('/static/js/dashboard.js')
record('GET static js','200', rv.status_code, rv.get_data(as_text=True), rv.status_code==200)

# Data integrity on main risk_results.csv
rdf, err = load_risk_results()
if rdf is None:
    record('risk_results integrity','file present', 0, err or '', False)
else:
    total = len(rdf)
    unique = rdf['invoice_no'].nunique()
    missing_id = int(rdf['invoice_no'].isna().sum())
    missing_score = int(rdf['final_risk_score'].isna().sum())
    min_score = float(rdf['final_risk_score'].min())
    max_score = float(rdf['final_risk_score'].max())
    dup = rdf['invoice_no'].duplicated().any()
    levels = set(rdf['risk_level'].unique().tolist())
    missing_exp = int(rdf['explanation'].isna().sum())
    passed = (total==20000 and unique==20000 and missing_id==0 and missing_score==0 and 0.0<=min_score<=1.0 and 0.0<=max_score<=1.0 and not dup and missing_exp==0 and levels.issubset({'Low','Medium','High'}))
    record('risk_results integrity','20k rows, unique IDs, no missing scores, scores 0-1, risk levels valid', total, f'unique={unique} min={min_score} max={max_score} missing_exp={missing_exp}', passed)

# Model evaluation presence
rv = client.get('/evaluation')
text = rv.get_data(as_text=True)
me_present = 'Accuracy' in text or 'ROC-AUC' in text
record('Evaluation content presence','metrics displayed', rv.status_code, text, rv.status_code==200 and me_present)

print(json.dumps(results, indent=2))

document.addEventListener('DOMContentLoaded', function () {
  const riskChartData = window.dashboardData && window.dashboardData.riskDistribution ? window.dashboardData.riskDistribution : null;
  const scoreChartData = window.dashboardData && window.dashboardData.riskScoreDistribution ? window.dashboardData.riskScoreDistribution : null;
  const modelChartData = window.modelChartData ? window.modelChartData : null;

  if (riskChartData && document.getElementById('riskDistributionChart')) {
    new Chart(document.getElementById('riskDistributionChart'), {
      type: 'bar',
      data: {
        labels: riskChartData.labels,
        datasets: [{
          label: 'Invoices',
          data: riskChartData.values,
          backgroundColor: ['#2e7d32', '#f9a825', '#d32f2f'],
          borderRadius: 8
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
      }
    });
  }

  if (scoreChartData && document.getElementById('riskScoreDistributionChart')) {
    new Chart(document.getElementById('riskScoreDistributionChart'), {
      type: 'line',
      data: {
        labels: scoreChartData.labels,
        datasets: [{
          label: 'Risk Score Distribution',
          data: scoreChartData.values,
          fill: false,
          borderColor: '#1a6ea8',
          backgroundColor: 'rgba(26, 110, 168, 0.2)',
          tension: 0.2,
          pointRadius: 3
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true } }
      }
    });
  }

  if (modelChartData && document.getElementById('modelComparisonChart')) {
    new Chart(document.getElementById('modelComparisonChart'), {
      type: 'radar',
      data: {
        labels: modelChartData.labels,
        datasets: modelChartData.datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          r: {
            beginAtZero: true,
            min: 0,
            max: 1,
            ticks: { stepSize: 0.2 }
          }
        }
      }
    });
  }
});

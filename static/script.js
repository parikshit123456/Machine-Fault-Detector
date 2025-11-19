// ------------------------------
// ✅ Standard sensor ranges
// ------------------------------
const ranges = {
  temp: [25, 45],
  pressure: [3, 5],
  flow: [140, 180],
  vibration: [0, 1],
  fillheight: [245, 255],
  power: [3, 5],
  co2: [380, 450],
  humidity: [45, 60]
};

// Global charts
let donutChart;
let barChart;
let logsChart;

// ------------------------------
// ✅ Initialize charts on page load
// ------------------------------
window.addEventListener("load", () => {

  // LOGS LINE CHART
  const logsCtx = document.getElementById("logsChart").getContext("2d");
logsChart = new Chart(logsCtx, {
  type: "line",
  data: {
    labels: [],
    datasets: [{
      label: "Error Probability (%)",
      data: [],
      borderWidth: 3,
      borderColor: "#0ee2d8ff",
      tension: 0.3,
      fill: false
    }]
  },
  options: {
    responsive: true,
    scales: {
      y: { 
        beginAtZero: true,
        max: 100,

        // ⭐ ADD THIS FOR Y-AXIS LABEL ⭐
        title: {
          display: true,
          text: "Error Probability (%)",
          font: { size: 14, weight: "bold" }
        }
      },
      x: {
        title: {
          display: true,
          text: "Timestamp",
          font: { size: 14, weight: "bold" }
        }
      }
    }
  }
});

  // DONUT CHART
  const donutCtx = document.getElementById('donutChart').getContext('2d');
  donutChart = new Chart(donutCtx, {
    type: 'doughnut',
    data: {
      labels: ['Safe', 'Error'],
      datasets: [{
        data: [100, 0],
        backgroundColor: ['#29c37a', '#e63946'],
        borderWidth: 0
      }]
    },
    options: {
      cutout: '70%',
      plugins: { legend: { display: false }, tooltip: { enabled: false } }
    }
  });

  // Center % text
  setDonutCenter("0%");

  // BAR CHART
  const barCtx = document.getElementById("barChart").getContext("2d");
  barChart = new Chart(barCtx, {
    type: "bar",
    data: {
      labels: Object.keys(ranges),
      datasets: [{
        label: "Sensor Values",
        data: Object.keys(ranges).map(_ => 0),
        backgroundColor: Object.keys(ranges).map(_ => "#29c37a")
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false }},
      scales: { y: { beginAtZero: true } }
    }
  });

  loadLogsChart();
});


// ------------------------------
// ✅ Donut center text
// ------------------------------
function setDonutCenter(text) {
  const c = document.getElementById("donutCenter");
  if (c) c.innerText = text;
}


// ------------------------------
// ✅ Predict Button Handler
// ------------------------------
async function onPredict() {

  const ids = ['temp','pressure','flow','vibration','fillheight','power','co2','humidity'];

  let allFilled = true;

  ids.forEach(id => document.getElementById(id).style.border = "1px solid #ccc");

  // Empty check
  ids.forEach(id => {
    const input = document.getElementById(id);
    if (!input.value) {
      input.style.border = "2px solid red";
      allFilled = false;
    }
  });

  if (!allFilled) {
    alert("⚠ Please fill all fields.");
    return;
  }

  // Create payload
  const payload = {};
  ids.forEach(id => payload[id] = Number(document.getElementById(id).value));

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await res.json();

    if (data.error) {
      alert("Server Error: " + data.error);
      return;
    }

    // Update donut chart
    const probPercent = Math.round((data.probability || 0) * 100);
    donutChart.data.datasets[0].data = [100 - probPercent, probPercent];
    donutChart.update();
    setDonutCenter(probPercent + "%");

    // Update error details
    document.getElementById("errCode").innerText = data.error_code;
    document.getElementById("errDesc").innerText = data.error_description;

    // Update bar chart
    updateBarChart(payload);

    // Update logs chart
    loadLogsChart();

  } catch (err) {
    alert("Communication Error: " + err);
  }
}


// ------------------------------
// ✅ BAR CHART UPDATE LOGIC (Corrected)
// ------------------------------
function updateBarChart(values) {
  const keys = Object.keys(ranges);

  const colors = keys.map(key => {
    const v = values[key];
    const [min, max] = ranges[key];
    return (v < min || v > max) ? "#e63946" : "#29c37a";
  });

  barChart.data.datasets[0].data = keys.map(key => values[key]);
  barChart.data.datasets[0].backgroundColor = colors;
  barChart.update();
}


// ------------------------------
// ✅ Load Logs into Line Chart
// ------------------------------
async function loadLogsChart() {
  try {
    const res = await fetch("/logs");
    const logs = await res.json();

    logsChart.data.labels = logs.map(l => l.timestamp);
    logsChart.data.datasets[0].data = logs.map(l => Math.round(l.probability * 100));

    logsChart.update();

  } catch (err) {
    console.log("Log fetch error:", err);
  }
}
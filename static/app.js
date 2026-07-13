let serverPayload = null;

function setStatus(state, label){
  const dot = document.getElementById('statusDot');
  const text = document.getElementById('statusText');
  dot.className = 'led-dot ' + state;
  text.innerText = label;
}

function setRail(activeStep, doneUpTo){
  document.querySelectorAll('.rail-step').forEach(el => {
    const n = parseInt(el.dataset.step);
    el.classList.remove('active','done');
    if(n < activeStep || n <= doneUpTo) el.classList.add('done');
    else if(n === activeStep) el.classList.add('active');
  });
}

function switchView(viewId){
  document.querySelectorAll('.view-section').forEach(el => el.classList.remove('active'));
  document.getElementById(viewId).classList.add('active');
}

function showError(msg){
  const banner = document.getElementById('errorBanner');
  banner.innerText = msg;
  banner.style.display = 'block';
}
function clearError(){
  document.getElementById('errorBanner').style.display = 'none';
}

/* ---------- Drag & drop / file select ---------- */
const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('file_live');
const fileArea = document.getElementById('fileArea');
const dzHint = document.getElementById('dzHint');

document.getElementById('browseBtn').addEventListener('click', () => fileInput.click());
dropzone.addEventListener('click', (e) => { if(e.target.id !== 'browseBtn') fileInput.click(); });

['dragenter','dragover'].forEach(evt => dropzone.addEventListener(evt, (e) => {
  e.preventDefault(); e.stopPropagation(); dropzone.classList.add('drag');
}));
['dragleave','drop'].forEach(evt => dropzone.addEventListener(evt, (e) => {
  e.preventDefault(); e.stopPropagation(); dropzone.classList.remove('drag');
}));
dropzone.addEventListener('drop', (e) => {
  const files = e.dataTransfer.files;
  if(files.length){ fileInput.files = files; renderFileChip(files[0]); }
});
fileInput.addEventListener('change', () => {
  if(fileInput.files.length) renderFileChip(fileInput.files[0]);
});

function renderFileChip(file){
  dzHint.style.display = 'none';
  fileArea.innerHTML = `
    <div class="file-chip">
      <span>${file.name}</span>
      <span class="x" id="clearFile" title="Remove">&#10005;</span>
    </div>`;
  document.getElementById('clearFile').addEventListener('click', (e) => {
    e.stopPropagation();
    fileInput.value = '';
    fileArea.innerHTML = '<button type="button" class="browse-btn" id="browseBtn">Browse files</button>';
    dzHint.style.display = 'block';
    document.getElementById('browseBtn').addEventListener('click', () => fileInput.click());
  });
}

/* ---------- Submit / ETL ---------- */
document.getElementById('uploadForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  clearError();

  if(!fileInput.files.length){
    showError('Select a .mat or .csv file before running the pipeline.');
    return;
  }

  const form = document.getElementById('uploadForm');
  const loader = document.getElementById('loader-upload');
  form.style.display = 'none';
  loader.style.display = 'block';
  setStatus('busy', 'Ingesting');

  const formData = new FormData();
  formData.append('file_live', fileInput.files[0]);

  try{
    const response = await fetch('/predict', { method: 'POST', body: formData });
    serverPayload = await response.json();

    if(serverPayload.error){
      form.style.display = 'block';
      loader.style.display = 'none';
      setStatus('fault', 'Ingest failed');
      showError(serverPayload.error);
      return;
    }

    document.getElementById('stat-raw').innerText = serverPayload.raw_rows;
    document.getElementById('stat-comp').innerText = serverPayload.compressed_rows;
    setRail(2, 1);
    setStatus('idle', 'Ready');
    switchView('view-etl');

  }catch(err){
    form.style.display = 'block';
    loader.style.display = 'none';
    setStatus('fault', 'Connection failed');
    showError('Connection failed: ' + err.message);
  }
});

document.getElementById('btn-run-ai').addEventListener('click', () => {
  document.getElementById('btn-run-ai').style.display = 'none';
  document.getElementById('loader-ai').style.display = 'block';
  setStatus('busy', 'Running inference');

  setTimeout(() => {
    setRail(3, 2);
    switchView('view-dashboard');
    requestAnimationFrame(() => {
      requestAnimationFrame(() => { renderDashboard(); });
    });
  }, 1000);
});

/* ---------- Dashboard ---------- */
function renderDashboard(){
  if(!serverPayload) return;
  const data = serverPayload;

  const banner = document.getElementById('result-banner');
  const card = document.getElementById('forecast-card');
  banner.innerHTML = `<span style="width:8px;height:8px;border-radius:50%;background:currentColor;display:inline-block;"></span>${data.diagnosis}`;

  if(data.is_faulty){
    banner.style.background = 'var(--danger-soft)';
    banner.style.color = 'var(--danger)';
    banner.style.border = '1px solid rgba(220,38,38,0.2)';
    document.getElementById('forecast-value').style.color = 'var(--danger)';
    card.style.borderLeftColor = 'var(--danger)';
    setStatus('fault', 'Fault detected');
  }else{
    banner.style.background = 'var(--success-soft)';
    banner.style.color = 'var(--success)';
    banner.style.border = '1px solid rgba(22,163,74,0.2)';
    document.getElementById('forecast-value').style.color = 'var(--success)';
    card.style.borderLeftColor = 'var(--success)';
    setStatus('ok', 'Nominal');
  }

  document.getElementById('forecast-value').innerText = data.days_remaining;

  document.getElementById('zoomSlider').value = 1000;
  document.getElementById('zoomValue').innerText = '1000 Hz';
  updateFreqChart(1000);
}

document.getElementById('zoomSlider').addEventListener('input', (e) => {
  const maxFreq = parseInt(e.target.value);
  document.getElementById('zoomValue').innerText = maxFreq + ' Hz';
  updateFreqChart(maxFreq);
});

function updateFreqChart(maxFreq){
  if(!serverPayload || !serverPayload.x_axis) return;
  let cutoffIndex = serverPayload.x_axis.findIndex(val => val > maxFreq);
  if(cutoffIndex === -1) cutoffIndex = serverPayload.x_axis.length;

  const slicedX = serverPayload.x_axis.slice(0, cutoffIndex);
  const slicedY = serverPayload.y_live.slice(0, cutoffIndex);
  const chartColor = serverPayload.is_faulty ? '#dc2626' : '#0284c7';

  drawChart('chartFreq', slicedX, slicedY, chartColor, 'Hz', true);
}

function drawChart(canvasId, xData, yData, color, xLabel, fillGradient){
  const ctx = document.getElementById(canvasId).getContext('2d');
  if(Chart.getChart(canvasId)){ Chart.getChart(canvasId).destroy(); }

  let bgFill = 'transparent';
  if(fillGradient){
    const gradient = ctx.createLinearGradient(0, 0, 0, 280);
    gradient.addColorStop(0, color + '22');
    gradient.addColorStop(1, color + '00');
    bgFill = gradient;
  }

  new Chart(ctx, {
    type: 'line',
    data: {
      labels: xData.map(val => Number(val).toFixed(1) + ' ' + xLabel),
      datasets: [{ data: yData, borderColor: color, borderWidth: 2, backgroundColor: bgFill, fill: fillGradient, pointRadius: 0, tension: 0.15 }]
    },
    options: {
      responsive: true, maintainAspectRatio: false, animation: { duration: 700, easing: 'easeOutQuart' },
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { color: '#f1f5f9', drawBorder: false }, ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 }, maxTicksLimit: 8 } },
        y: { grid: { color: '#f1f5f9', drawBorder: false }, ticks: { color: '#64748b', font: { family: 'JetBrains Mono', size: 10 } } }
      },
      interaction: { intersect: false, mode: 'index' }
    }
  });
}
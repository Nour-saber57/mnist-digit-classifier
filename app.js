let model;
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const statusEl = document.getElementById('status');
const predictionEl = document.getElementById('prediction');
const barsEl = document.getElementById('bars');
const predictBtn = document.getElementById('predictBtn');
const clearBtn = document.getElementById('clearBtn');

let drawing = false;

function resetCanvas() {
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 22;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = 'white';
  predictionEl.textContent = 'Prediction: —';
  barsEl.innerHTML = '';
}

function getPointerPos(event) {
  const rect = canvas.getBoundingClientRect();
  const clientX = event.touches ? event.touches[0].clientX : event.clientX;
  const clientY = event.touches ? event.touches[0].clientY : event.clientY;
  return {
    x: (clientX - rect.left) * (canvas.width / rect.width),
    y: (clientY - rect.top) * (canvas.height / rect.height)
  };
}

function startDraw(event) {
  event.preventDefault();
  drawing = true;
  const p = getPointerPos(event);
  ctx.beginPath();
  ctx.moveTo(p.x, p.y);
}

function draw(event) {
  if (!drawing) return;
  event.preventDefault();
  const p = getPointerPos(event);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
}

function stopDraw(event) {
  event.preventDefault();
  drawing = false;
}

async function loadModel() {
  try {
    statusEl.textContent = 'Loading model weights...';

    // Build the SAME architecture manually in TensorFlow.js.
    // This avoids a common Keras 3 -> TFJS deserialization problem on Netlify/browser.
    const builtModel = tf.sequential();
    builtModel.add(tf.layers.flatten({ inputShape: [28, 28] }));
    builtModel.add(tf.layers.dense({ units: 120, activation: 'relu' }));
    builtModel.add(tf.layers.dense({ units: 120, activation: 'relu' }));
    builtModel.add(tf.layers.dense({ units: 10, activation: 'softmax' }));

    // Create variables by running one dummy prediction.
    builtModel.predict(tf.zeros([1, 28, 28])).dispose();

    // Load only the weights from the exported model.json + .bin file.
    const response = await fetch('./model/model.json', { cache: 'no-store' });
    if (!response.ok) {
      throw new Error(`Could not fetch ./model/model.json. HTTP ${response.status}`);
    }
    const modelJson = await response.json();
    const weights = await tf.io.loadWeights(modelJson.weightsManifest, './model/');

    builtModel.setWeights([
      weights['sequential/dense/kernel'],
      weights['sequential/dense/bias'],
      weights['sequential/dense_1/kernel'],
      weights['sequential/dense_1/bias'],
      weights['sequential/dense_2/kernel'],
      weights['sequential/dense_2/bias']
    ]);

    model = builtModel;
    statusEl.textContent = 'Model loaded ✅ Draw a digit.';
    predictBtn.disabled = false;
  } catch (error) {
    console.error('Model loading error:', error);
    statusEl.textContent = 'Model loading failed. Open browser Console for the exact error.';
    predictBtn.disabled = true;
  }
}

function getDigitBoundingBox() {
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const data = imageData.data;
  let minX = canvas.width, minY = canvas.height, maxX = -1, maxY = -1;

  for (let y = 0; y < canvas.height; y++) {
    for (let x = 0; x < canvas.width; x++) {
      const idx = (y * canvas.width + x) * 4;
      const brightness = data[idx]; // white digit on black background, R channel is enough
      if (brightness > 20) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  if (maxX < 0) return null;
  return { minX, minY, maxX, maxY, width: maxX - minX + 1, height: maxY - minY + 1 };
}

function preprocessCanvas() {
  return tf.tidy(() => {
    const box = getDigitBoundingBox();
    if (!box) {
      return tf.zeros([1, 28, 28]);
    }

    // MNIST digits are centered and occupy most of a 28x28 image.
    // The user canvas is large, so we crop the drawn digit first, resize it,
    // then place it in the center of a 28x28 black image.
    const temp = document.createElement('canvas');
    temp.width = 28;
    temp.height = 28;
    const tctx = temp.getContext('2d');
    tctx.fillStyle = 'black';
    tctx.fillRect(0, 0, 28, 28);
    tctx.imageSmoothingEnabled = true;
    tctx.imageSmoothingQuality = 'high';

    const margin = 18;
    let sx = Math.max(0, box.minX - margin);
    let sy = Math.max(0, box.minY - margin);
    let sw = Math.min(canvas.width - sx, box.width + 2 * margin);
    let sh = Math.min(canvas.height - sy, box.height + 2 * margin);

    const scale = 20 / Math.max(sw, sh);
    const dw = sw * scale;
    const dh = sh * scale;
    const dx = (28 - dw) / 2;
    const dy = (28 - dh) / 2;

    tctx.drawImage(canvas, sx, sy, sw, sh, dx, dy, dw, dh);

    let img = tf.browser.fromPixels(temp, 1); // [28, 28, 1]
    img = img.toFloat().div(255.0);
    img = img.squeeze([2]).expandDims(0); // [1, 28, 28]

    // Match your training exactly: tf.keras.utils.normalize(X_train, axis=1)
    const norm = img.square().sum(1, true).sqrt().add(1e-7);
    img = img.div(norm);

    return img;
  });
}

async function predictDigit() {
  if (!model) return;
  const input = preprocessCanvas();
  const output = model.predict(input);
  const probs = await output.data();
  input.dispose();
  output.dispose();

  let bestIndex = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[bestIndex]) bestIndex = i;
  }

  predictionEl.textContent = `Prediction: ${bestIndex}`;
  renderBars(probs);
}

function renderBars(probs) {
  barsEl.innerHTML = '';
  probs.forEach((p, i) => {
    const row = document.createElement('div');
    row.className = 'bar-row';
    const label = document.createElement('strong');
    label.textContent = i;
    const bg = document.createElement('div');
    bg.className = 'bar-bg';
    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.width = `${(p * 100).toFixed(1)}%`;
    const value = document.createElement('span');
    value.textContent = `${(p * 100).toFixed(1)}%`;
    bg.appendChild(fill);
    row.appendChild(label);
    row.appendChild(bg);
    row.appendChild(value);
    barsEl.appendChild(row);
  });
}

canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDraw);
canvas.addEventListener('mouseleave', stopDraw);
canvas.addEventListener('touchstart', startDraw);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDraw);

predictBtn.addEventListener('click', predictDigit);
clearBtn.addEventListener('click', resetCanvas);

resetCanvas();
loadModel();

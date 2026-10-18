import {summarizeScores} from './inference-utils.mjs?v=20261018-face-oe';

const relativeURL = path => new URL(path, import.meta.url).href;
let sessionPromise;

async function fetchModel(config, progress) {
  const modelURL = new URL(relativeURL(config.model));
  modelURL.searchParams.set('v', config.modelSha256);
  const response = await fetch(modelURL.href);
  if (!response.ok) throw new Error('模型下载失败，请检查网络后重试。');
  let bytes;
  if (response.body?.getReader) {
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const {value, done} = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      progress({stage: 'download', received, total: config.modelBytes});
    }
    bytes = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  } else {
    bytes = new Uint8Array(await response.arrayBuffer());
  }
  if (bytes.byteLength !== config.modelBytes) throw new Error('模型文件不完整，请重新加载。');
  if (crypto.subtle) {
    const digest = await crypto.subtle.digest('SHA-256', bytes);
    const hex = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
    if (hex !== config.modelSha256) throw new Error('模型版本校验失败，请刷新后重新加载。');
  }
  return bytes;
}

export function loadSession(config, progress = () => {}) {
  if (!sessionPromise) {
    sessionPromise = (async () => {
      if (!window.ort || typeof WebAssembly === 'undefined') throw new Error('当前浏览器暂不支持识别，请使用较新的 Chrome、Edge、Safari 或 Firefox。');
      // GitHub Pages 不要求 cross-origin isolation，因此使用单线程 WASM。
      window.ort.env.wasm.numThreads = 1;
      window.ort.env.wasm.proxy = false;
      window.ort.env.wasm.wasmPaths = relativeURL('./vendor/');
      const bytes = await fetchModel(config, progress);
      progress({stage: 'initialize'});
      const session = await window.ort.InferenceSession.create(bytes, {executionProviders: ['wasm'], graphOptimizationLevel: 'all'});
      progress({stage: 'ready'});
      return session;
    })().catch(error => { sessionPromise = undefined; throw error; });
  }
  return sessionPromise;
}

export async function imageTensor(file, config) {
  let bitmap;
  try { bitmap = await createImageBitmap(file, {imageOrientation: 'from-image'}); }
  catch { throw new Error('这张图片无法打开，请换成 JPG、PNG 或 WebP 图片。'); }
  try {
    if (!bitmap.width || !bitmap.height || bitmap.width * bitmap.height > 40_000_000) {
      throw new Error('图片尺寸太大，请缩小到 4000 万像素以内再试。');
    }
    const [left, top, right, bottom] = config.crop;
    const x1 = Math.round(bitmap.width * left), y1 = Math.round(bitmap.height * top);
    const x2 = Math.round(bitmap.width * (1 - right)), y2 = Math.round(bitmap.height * (1 - bottom));
    const sourceWidth = x2-x1, sourceHeight = y2-y1;
    if (sourceWidth < 1 || sourceHeight < 1) throw new Error('图片尺寸太小，无法识别。');
    const size = config.inputSize;
    const ratio = Math.min(size/sourceWidth, size/sourceHeight);
    const width = Math.max(1, Math.round(sourceWidth * ratio));
    const height = Math.max(1, Math.round(sourceHeight * ratio));
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext('2d', {willReadFrequently: true});
    if (!ctx) throw new Error('无法读取图片，请更换浏览器重试。');
    ctx.fillStyle = `rgb(${config.background.join(',')})`;
    ctx.fillRect(0, 0, size, size);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(bitmap, x1, y1, sourceWidth, sourceHeight,
      Math.floor((size-width)/2), Math.floor((size-height)/2), width, height);
    const pixels = ctx.getImageData(0, 0, size, size).data;
    const plane = size*size;
    const data = new Float32Array(plane*3);
    for (let pixel = 0; pixel < plane; pixel++) {
      for (let channel = 0; channel < 3; channel++) {
        data[channel*plane+pixel] = (pixels[pixel*4+channel]/255-config.mean[channel])/config.std[channel];
      }
    }
    return new window.ort.Tensor('float32', data, [1, 3, size, size]);
  } finally { bitmap.close(); }
}

export async function recognize(file, config, progress) {
  const session = await loadSession(config, progress);
  const tensor = await imageTensor(file, config);
  let output;
  try {
    output = await session.run({images: tensor});
    return summarizeScores(output.logits.data, config);
  } finally {
    tensor.dispose();
    if (output) for (const value of Object.values(output)) value.dispose();
  }
}

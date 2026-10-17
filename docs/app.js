import {loadSession, recognize} from './inference.js';

const $ = id => document.getElementById(id);
const asset = path => new URL(path, import.meta.url).href;
const percent = value => `${(value * 100).toFixed(1)}%`;
const descriptions = {
  '奶史': '一圈一圈的快乐，堆成了最有辨识度的模样。',
  '奶狗': '耳朵软软的，快乐也是。今天也要摇摇尾巴！',
  '奶猫': '安静地可爱一下，偶尔再偷偷卖个萌。',
  '奶粉': '粉粉嫩嫩，连空气都变得软乎乎。',
  '奶蛋': '圆滚滚的小太阳，把今天的好心情打包。',
  '奶蛙': '肚皮一挺，快乐出击。奶门的经典成员来了！',
  '奶鸡': '精神抖擞的小鸡冠，今天也要元气满满。',
  '奶鼠': '大大的耳朵，小小的身体，装满了好奇心。',
  '奶龙': '熟悉的奶龙登场，这份快乐必须认领。',
  '小奶': '小小一只，不用做什么，出现就已经很治愈。',
  '豆奶': '奶家人的特别来宾，把奇妙和可爱一起带来。'
};
let config, configPromise, selectedFile, previewURL, busy = false, imageVersion = 0, modelReady = false;

function errorMessage(message = '') {
  $('upload-error').textContent = message;
  $('upload-error').hidden = !message;
}

function refreshButton() {
  $('recognize-button').disabled = !selectedFile || busy || !config;
  $('recognize-label').textContent = busy ? '正在寻找你的奶家人…' : '开始识别';
  $('recognize-arrow').className = busy ? 'spinner' : '';
  $('recognize-arrow').textContent = busy ? '' : '↗';
  $('result-panel').setAttribute('aria-busy', String(busy));
  $('try-example').disabled = busy || !config;
  for (const button of document.querySelectorAll('.family-tile')) button.disabled = busy;
}

function progress(state) {
  const meter = $('model-progress');
  $('retry-model').hidden = true;
  if (state.stage === 'download') {
    meter.hidden = false;
    meter.value = Math.min(90, state.received/state.total*90);
    $('runtime-status').textContent = `正在准备模型 · ${(state.received/1024/1024).toFixed(1)} / ${(state.total/1024/1024).toFixed(1)} MB`;
  } else if (state.stage === 'initialize') {
    meter.value = 95;
    $('runtime-status').textContent = '下载完成，马上就好…';
  } else if (state.stage === 'ready') {
    modelReady = true;
    meter.value = 100;
    meter.hidden = true;
    $('runtime-dot').className = 'runtime-dot ready';
    $('runtime-status').textContent = '模型已就绪 · 图片仅在本机处理';
  }
}

function modelError(error) {
  modelReady = false;
  $('runtime-dot').className = 'runtime-dot failed';
  $('runtime-status').textContent = error.message || '模型加载失败，请重试。';
  $('retry-model').hidden = false;
  $('model-progress').hidden = true;
}

function chooseFile(file) {
  if (!file) return;
  const allowed = /^(image\/(jpeg|png|webp|avif|bmp|gif))$/i.test(file.type) ||
    (!file.type && /\.(jpe?g|png|webp|avif|bmp|gif)$/i.test(file.name));
  if (!allowed) { errorMessage('请选择 JPG、PNG、WebP 等图片文件，暂不支持此格式。'); return; }
  if (!file.size || file.size > 20*1024*1024) { errorMessage('请选择一张非空且小于 20 MB 的图片。'); return; }
  imageVersion++;
  selectedFile = file;
  errorMessage();
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = URL.createObjectURL(file);
  $('input-preview').src = previewURL;
  $('file-name').textContent = file.name;
  $('upload-placeholder').hidden = true;
  $('upload-preview').hidden = false;
  $('result-content').hidden = true;
  $('result-empty').hidden = false;
  $('result-panel').classList.remove('is-unknown');
  refreshButton();
}

function showResult(result) {
  const {unknown, display, topMatches, confidence} = result;
  $('result-panel').classList.toggle('is-unknown', unknown);
  $('result-kicker').textContent = unknown ? '暂时没找到奶家人' : '奶家人认证成功';
  $('match-badge').textContent = `${unknown ? '最高匹配 ' : ''}${percent(confidence)}${unknown ? '' : ' 匹配'}`;
  $('result-name').textContent = unknown ? config.unknownMessage : display.name;
  $('result-description').textContent = unknown ? '先让小奶陪你一会儿。也可以换张主体更清晰的图片再试试。' : descriptions[display.name];
  $('result-image').src = asset(display.image);
  $('result-image').alt = `${display.name}展示图`;
  const list = $('match-list');
  list.replaceChildren();
  for (const item of topMatches) {
    const li = document.createElement('li'); li.className = 'match-row';
    const name = document.createElement('span'); name.textContent = item.name;
    const track = document.createElement('div'); track.className = 'match-track';
    const fill = document.createElement('div'); fill.className = 'match-fill'; fill.style.width = percent(item.probability);
    const value = document.createElement('span'); value.className = 'match-value'; value.textContent = percent(item.probability);
    track.append(fill); li.append(name, track, value); list.append(li);
  }
  $('result-empty').hidden = true;
  $('result-content').hidden = false;
}

async function runRecognition() {
  if (!selectedFile || busy || !config) return;
  const version = imageVersion, file = selectedFile;
  busy = true; errorMessage(); refreshButton();
  try {
    const result = await recognize(file, config, progress);
    if (version === imageVersion) showResult(result);
  } catch (error) {
    if (version === imageVersion) errorMessage(error.message || '识别失败，请换一张图片重试。');
    if (!modelReady) modelError(error);
  } finally { busy = false; refreshButton(); }
}

async function useExample(item) {
  if (busy) return;
  try {
    const response = await fetch(asset(item.image));
    if (!response.ok) throw new Error('示例图片加载失败，请稍后重试。');
    const blob = await response.blob();
    chooseFile(new File([blob], `${item.name}示例.webp`, {type: 'image/webp'}));
    $('recognizer').scrollIntoView({behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start'});
    await runRecognition();
  } catch (error) { errorMessage(error.message); }
}

function renderFamily() {
  const grid = $('family-grid'); grid.replaceChildren();
  for (const item of config.classes) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'family-tile';
    button.setAttribute('aria-label', `用${item.name}展示图试试识别`);
    const image = document.createElement('img'); image.src = asset(item.image); image.alt = item.name; image.loading = 'lazy';
    const caption = document.createElement('div'); caption.className = 'tile-caption';
    const name = document.createElement('strong'); name.textContent = item.name;
    const arrow = document.createElement('span'); arrow.textContent = '↗'; arrow.setAttribute('aria-hidden', 'true');
    caption.append(name, arrow); button.append(image, caption);
    button.addEventListener('click', () => useExample(item)); grid.append(button);
  }
}

async function init() {
  try {
    if (!configPromise) configPromise = fetch(asset('./model-config.json')).then(response => {
      if (!response.ok) throw new Error('识别配置加载失败，请刷新页面后重试。');
      return response.json();
    }).catch(error => { configPromise = undefined; throw error; });
    config = await configPromise;
    renderFamily(); refreshButton();
    $('runtime-dot').className = 'runtime-dot';
    $('runtime-status').textContent = '正在准备识别模型…';
    await loadSession(config, progress);
  } catch (error) { modelError(error); }
}

$('image-input').addEventListener('change', event => { chooseFile(event.target.files[0]); event.target.value = ''; });
$('recognize-button').addEventListener('click', runRecognition);
$('try-example').addEventListener('click', () => { if (config) useExample(config.classes.find(item => item.name === '奶蛙')); });
$('retry-model').addEventListener('click', init);
$('input-preview').addEventListener('error', () => { selectedFile = undefined; errorMessage('图片无法预览，请选择可以正常打开的 JPG、PNG 或 WebP 图片。'); refreshButton(); });
const dropZone = $('drop-zone');
for (const type of ['dragenter','dragover']) dropZone.addEventListener(type, event => { event.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', event => { if (!dropZone.contains(event.relatedTarget)) dropZone.classList.remove('drag-over'); });
dropZone.addEventListener('drop', event => { event.preventDefault(); dropZone.classList.remove('drag-over'); chooseFile(event.dataTransfer.files[0]); });
window.addEventListener('beforeunload', () => { if (previewURL) URL.revokeObjectURL(previewURL); });
$('try-example').disabled = true;
init();

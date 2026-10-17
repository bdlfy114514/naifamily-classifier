/** 与 DOM 无关的分类后处理，可直接用 Node 运行回归测试。 */
export function softmax(logits) {
  if (!logits.length || Array.from(logits).some(value => !Number.isFinite(value))) {
    throw new Error('模型输出无效，请重新选择图片。');
  }
  const maximum = Math.max(...logits);
  const values = Array.from(logits, value => Math.exp(value - maximum));
  const sum = values.reduce((a, b) => a + b, 0);
  return values.map(value => value / sum);
}

export function summarizeScores(logits, config) {
  if (logits.length !== config.classes.length || (config.classes.length < 2 || !Number.isFinite(config.minTopScoreRatio) || config.minTopScoreRatio < 1)) {
    throw new Error('模型与类别配置不一致，请刷新页面。');
  }
  const probabilities = softmax(logits);
  const ranking = probabilities.map((probability, index) => ({...config.classes[index], index, probability}))
    .sort((a, b) => b.probability - a.probability || a.index - b.index);
  const unknown = ranking[0].probability <= ranking[1].probability * config.minTopScoreRatio;
  const display = unknown ? config.classes.find(item => item.name === config.fallbackClass) : ranking[0];
  if (!display) throw new Error('缺少展示图配置。');
  return {unknown, display, topMatches: ranking.slice(0, 3), confidence: ranking[0].probability};
}

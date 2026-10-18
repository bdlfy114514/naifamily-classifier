import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, existsSync} from 'node:fs';
import {createHash} from 'node:crypto';
import {softmax, summarizeScores} from '../docs/inference-utils.mjs';

const config = JSON.parse(readFileSync(new URL('../docs/model-config.json', import.meta.url), 'utf8'));

test('softmax handles large logits without overflow', () => {
  const probabilities = softmax([10000,10001,10002]);
  assert.ok(probabilities.every(Number.isFinite));
  assert.ok(Math.abs(probabilities.reduce((a,b)=>a+b)-1) < 1e-12);
  assert.ok(probabilities[2]>probabilities[1]);
});

test('high match returns the winning class and its own show image', () => {
  const logits = config.classes.map(item => item.name === '奶蛙' ? 9 : -2);
  const result = summarizeScores(logits, config);
  assert.equal(result.unknown, false);
  assert.equal(result.display.name, '奶蛙');
  assert.equal(result.display.image, config.classes.find(item=>item.name==='奶蛙').image);
  assert.equal(result.topMatches.length, 3);
});

test('tied matches return the exact requested message configuration and 小奶 image', () => {
  const result = summarizeScores(config.classes.map(()=>0), config);
  assert.equal(result.unknown, true);
  assert.equal(config.unknownMessage, 'ops! 您上传的的图片中似乎没有奶家人');
  assert.equal(result.display.name, '小奶');
  assert.equal(result.display.image, config.classes.find(item=>item.name==='小奶').image);
  assert.equal(result.topMatches[0].probability, result.topMatches[1].probability);
});

test('invalid model outputs fail instead of inventing a prediction', () => {
  assert.throws(()=>softmax([Infinity,0]));
  assert.throws(()=>summarizeScores([0],config));
  assert.throws(()=>summarizeScores(config.classes.map(()=>NaN),config));
});

test('acceptance requires both strict 1.5 ratio and strict 25 percent confidence', () => {
  assert.equal(config.minTopScoreRatio, 1.5);
  assert.equal(config.minConfidence, 0.25);
  assert.equal('unknownThreshold' in config, false);
  for (const [first, second, expectedUnknown] of [
    [0.30, 0.15, false],
    [0.24, 0.10, true], // Clear winner but below 25%.
    [0.25, 0.10, true], // Exactly 25% is rejected.
    [0.2501, 0.10, false],
    [0.30, 0.21, true], // Above 25% but insufficient lead. // Below the old 40% cutoff, but a clear winner.
    [0.60, 0.40, true], // Exactly 1.5 times is rejected.
    [0.59, 0.40, true], // High absolute score is insufficient.
    [0.61, 0.39, false],
    [0.40, 0.40, true],
  ]) {
    const remaining = Math.max(0, 1-first-second)/(config.classes.length-2);
    const logits = config.classes.map((_, index) => Math.log(
      Math.max(Number.MIN_VALUE, index === 0 ? first : index === 1 ? second : remaining)));
    const result = summarizeScores(logits, config);
    assert.equal(result.unknown, expectedUnknown, `${first} vs ${second}`);
    assert.equal(result.display.name, expectedUnknown ? '小奶' : config.classes[0].name);
  }
});

test('published assets match the model and version metadata', () => {
  assert.equal(new Set(config.classes.map(item=>item.name)).size, 11);
  for (const item of config.classes) assert.ok(existsSync(new URL(`../docs/${item.image}`, import.meta.url)));
  const model = readFileSync(new URL(`../docs/${config.model}`, import.meta.url));
  assert.equal(model.length, config.modelBytes);
  assert.equal(createHash('sha256').update(model).digest('hex'), config.modelSha256);
  const runtime = JSON.parse(readFileSync(new URL('../docs/vendor/version.json', import.meta.url), 'utf8'));
  assert.equal(runtime.version, '1.23.2');
  for (const [file, hash] of Object.entries(runtime.sha256)) {
    assert.equal(createHash('sha256').update(readFileSync(new URL(`../docs/vendor/${file}`, import.meta.url))).digest('hex'), hash);
  }
});

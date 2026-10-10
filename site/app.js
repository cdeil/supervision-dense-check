"use strict";
const state = {count: 1000, classes: 1, missing: false, cap: 1000};
const $ = id => document.getElementById(id);
let report, generation = 0;
const imageCache = new Map();
const format = value => value.toLocaleString("en-US");
const percent = value => (value * 100).toFixed(2) + "%";
for (const count of [25, 100, 101, 350, 1000]) {
  const button = document.createElement("button");
  button.type = "button"; button.textContent = format(count);
  button.dataset.count = count; button.setAttribute("aria-pressed", count === state.count);
  button.addEventListener("click", () => {state.count = count; render().catch(fail);});
  $("counts").append(button);
}
for (const id of ["classes", "missing", "cap"]) $(id).addEventListener("change", () => {
  state[id] = id === "missing" ? $(id).value === "true" : Number($(id).value);
  render().catch(fail);
});
function fail(error) {$("status").textContent = "Could not load the measured results: " + error.message;}
function loadImage(source) {
  if (!imageCache.has(source)) imageCache.set(source, new Promise((resolve, reject) => {
    const image = new Image(); image.onload = () => resolve(image);
    image.onerror = () => reject(new Error("image unavailable")); image.src = source;
  }));
  return imageCache.get(source);
}
function draw(id, scene, image, cap) {
  const context = $(id).getContext("2d"); context.clearRect(0, 0, 1024, 768);
  context.drawImage(image, 0, 0);
  const ranks = Array(scene.classes).fill(0);
  const predicted = new Set(scene.predictions.map(box => box.join(",")));
  let scored = 0;
  scene.boxes.forEach((box, index) => {
    const present = predicted.has(box.join(","));
    const enters = present && ++ranks[index % scene.classes] <= cap;
    if (enters) scored++;
    context.strokeStyle = !present ? "#aaa5af" : enters ? "#168777" : "#e49332";
    context.lineWidth = 1.7; context.setLineDash(present ? [] : [4, 3]);
    context.strokeRect(box[0] - 2, box[1] - 2, box[2] - box[0] + 4, box[3] - box[1] + 4);
  });
  context.setLineDash([]); return scored;
}
async function render() {
  if (!report) return;
  const current = ++generation;
  document.querySelectorAll("[data-count]").forEach(button => button.setAttribute("aria-pressed", Number(button.dataset.count) === state.count));
  const matches = item => item.count === state.count && item.classes === state.classes && item.missing === state.missing;
  const before = report.before.cases.find(matches), after = report.after.cases.find(matches);
  const baseline = before.measurements[0], selected = after.measurements.find(item => item.cap === state.cap);
  const image = await loadImage(after.image); if (current !== generation) return;
  const beforeScored = draw("before", before, image, 100), afterScored = draw("after", after, image, state.cap);
  $("before-score").textContent = percent(baseline.ap50_95);
  $("after-score").textContent = percent(selected.ap50_95);
  $("after-title").textContent = "Selected cap: " + format(state.cap);
  const detail = scored => `${format(scored)} enter matching · ${format(after.detected - scored)} capped · ${format(after.count - after.detected)} missed`;
  $("before-detail").textContent = detail(beforeScored); $("after-detail").textContent = detail(afterScored);
  $("status").textContent = `Detector found ${format(after.detected)} / ${format(after.count)} stars. ` + (beforeScored < afterScored ? `The larger cap admits ${format(afterScored - beforeScored)} more predictions.` : "Both settings admit the same predictions.");
  $("ap50").textContent = `AP50: ${percent(baseline.ap50)} → ${percent(selected.ap50)}`;
}
fetch("results.json").then(response => {if (!response.ok) throw new Error(response.status); return response.json();}).then(data => {
  report = data; $("versions").textContent = "Pinned baseline: supervision " + data.before.versions.supervision + "; candidate: " + data.after.versions.supervision + ". Package versions and image hashes are recorded in the JSON.";
  return render();
}).catch(fail);

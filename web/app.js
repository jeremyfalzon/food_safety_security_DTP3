const state = {
  model: null,
  values: {},
  timer: null,
  requestId: 0,
};

const $ = (id) => document.getElementById(id);

const BANDS = [
  { max: 2.5, key: "low", label: "Low" },
  { max: 10, key: "moderate", label: "Moderate" },
  { max: 25, key: "high", label: "High" },
  { max: Infinity, key: "very-high", label: "Very high" },
];

function bandFor(value) {
  return BANDS.find((band) => value < band.max);
}

function inputValue(feature, value) {
  if (feature.key === "gdp") return String(Math.round(value));
  if (feature.key === "stability") return value.toFixed(2);
  return value.toFixed(1);
}

function displayValue(feature, value) {
  if (feature.key === "gdp") return Math.round(value).toLocaleString("en-US");
  if (feature.key === "stability") return value.toFixed(2);
  return value.toFixed(1);
}

function updateFill(range, feature, value) {
  const percent = ((value - feature.min) / (feature.max - feature.min)) * 100;
  range.style.setProperty("--fill", `${percent}%`);
}

function renderStats() {
  const { metrics } = state.model;
  $("stat-countries").textContent = metrics.n_countries;
  $("stat-adjr2").textContent = metrics.test.adjusted_r2.toFixed(2);
  $("metric-adjr2").textContent = metrics.test.adjusted_r2.toFixed(3);
  $("metric-adjr2-note").textContent = `${metrics.n_test} holdout countries`;
  $("metric-r2").textContent = metrics.test.r2.toFixed(3);
  $("metric-rmse").textContent = metrics.test.rmse.toFixed(1);
}

function renderPresets() {
  const select = $("country-preset");
  const countries = [...state.model.countries].sort((a, b) => a.name.localeCompare(b.name));
  for (const country of countries) {
    const option = document.createElement("option");
    option.value = country.name;
    option.textContent = country.name;
    select.appendChild(option);
  }
}

function renderControls() {
  const container = $("feature-controls");
  container.textContent = "";

  for (const feature of state.model.features) {
    const control = document.createElement("div");
    control.className = "control";

    const head = document.createElement("div");
    head.className = "control-head";

    const label = document.createElement("label");
    label.htmlFor = `number-${feature.key}`;
    label.textContent = feature.label;
    const unit = document.createElement("span");
    unit.className = "unit";
    unit.textContent = feature.unit;
    label.appendChild(unit);

    const number = document.createElement("input");
    number.type = "number";
    number.id = `number-${feature.key}`;
    number.min = feature.min;
    number.max = feature.max;
    number.step = feature.step;
    number.inputMode = "decimal";

    head.append(label, number);

    const range = document.createElement("input");
    range.type = "range";
    range.id = `range-${feature.key}`;
    range.min = feature.min;
    range.max = feature.max;
    range.step = feature.step;
    range.setAttribute("aria-label", feature.label);

    const hint = document.createElement("p");
    hint.className = "control-hint";
    hint.textContent = feature.hint;

    control.append(head, range, hint);
    container.appendChild(control);

    range.addEventListener("input", () => {
      applyValue(feature, Number(range.value), "range");
    });

    number.addEventListener("change", () => {
      const parsed = Number(number.value);
      if (!Number.isFinite(parsed)) {
        number.value = inputValue(feature, state.values[feature.key]);
        return;
      }
      applyValue(feature, parsed, "number");
    });
  }
}

function applyValue(feature, raw, source) {
  const value = Math.min(feature.max, Math.max(feature.min, raw));
  state.values[feature.key] = value;

  const range = $(`range-${feature.key}`);
  const number = $(`number-${feature.key}`);
  range.value = value;
  number.value = inputValue(feature, value);
  updateFill(range, feature, value);

  if (source === "range") {
    $("country-preset").value = "";
  }

  schedulePrediction();
}

function syncInputs(presetName = "") {
  for (const feature of state.model.features) {
    const value = state.values[feature.key];
    const range = $(`range-${feature.key}`);
    range.value = value;
    $(`number-${feature.key}`).value = inputValue(feature, value);
    updateFill(range, feature, value);
  }
  $("country-preset").value = presetName;
}

function renderCoefficients() {
  const list = $("coef-list");
  list.textContent = "";
  const max = Math.max(...state.model.features.map((f) => Math.abs(f.coef_std))) || 1;

  for (const feature of state.model.features) {
    const negative = feature.coef_std < 0;
    const width = (Math.abs(feature.coef_std) / max) * 50;
    const points = Math.abs(feature.coef_std).toFixed(1);

    const row = document.createElement("li");
    row.className = "coef-row";

    const head = document.createElement("div");
    head.className = "coef-head";
    const name = document.createElement("span");
    name.className = "coef-name";
    name.textContent = feature.label;
    const value = document.createElement("span");
    value.className = "coef-value";
    value.textContent = `${feature.coef_std > 0 ? "+" : "−"}${Math.abs(feature.coef_std).toFixed(2)}`;
    head.append(name, value);

    const track = document.createElement("div");
    track.className = "coef-track";
    const bar = document.createElement("span");
    bar.className = `coef-bar ${negative ? "negative" : "positive"}`;
    bar.style.width = `${width}%`;
    track.appendChild(bar);

    const hint = document.createElement("p");
    hint.className = "coef-hint";
    hint.textContent = `A one-standard-deviation increase is associated with ${points} percentage points ${negative ? "lower" : "higher"} undernourishment.`;

    row.append(head, track, hint);
    list.appendChild(row);
  }
}

function renderTable() {
  const body = $("country-body");
  body.textContent = "";
  const countries = [...state.model.countries].sort((a, b) => a.name.localeCompare(b.name));
  const fragment = document.createDocumentFragment();

  for (const country of countries) {
    const row = document.createElement("tr");
    row.dataset.name = country.name.toLowerCase();

    const cells = [
      { text: country.name, className: "country" },
      { text: Math.round(country.values.gdp).toLocaleString("en-US"), className: "num" },
      { text: country.values.stability.toFixed(2), className: "num" },
      { text: `${country.values.sanitation.toFixed(1)}%`, className: "num" },
      { text: `${country.values.clean_fuels.toFixed(1)}%`, className: "num" },
      { text: `${country.actual.toFixed(1)}%`, className: "num" },
      { text: `${country.predicted.toFixed(1)}%`, className: "num predicted" },
    ];

    for (const cell of cells) {
      const td = document.createElement("td");
      td.className = cell.className;
      td.textContent = cell.text;
      row.appendChild(td);
    }

    fragment.appendChild(row);
  }

  body.appendChild(fragment);
  $("table-meta").textContent = `${countries.length} countries · predicted values from the fitted regression`;
}

function renderPrediction(value) {
  const band = bandFor(value);
  $("prediction").textContent = value.toFixed(1);
  $("gauge-fill").style.width = `${Math.min(100, Math.max(0, value))}%`;
  $("gauge-marker").style.left = `${Math.min(100, Math.max(0, value))}%`;

  const chip = $("band-chip");
  chip.textContent = band.label;
  chip.dataset.band = band.key;

  if (value >= 0.5) {
    $("result-share").textContent = `Roughly 1 in ${Math.max(2, Math.round(100 / value))} people.`;
  } else {
    $("result-share").textContent = "Fewer than 1 in 200 people.";
  }

  const total = state.model.countries.length;
  const lower = state.model.countries.filter((c) => c.actual < value).length;
  if (lower === 0) {
    $("result-note").textContent = `Lower than every one of the ${total} countries in the sample.`;
  } else if (lower === total) {
    $("result-note").textContent = `Higher than every one of the ${total} countries in the sample.`;
  } else {
    $("result-note").textContent = `Higher than ${lower} of the ${total} countries in the sample.`;
  }
}

function schedulePrediction() {
  clearTimeout(state.timer);
  state.timer = setTimeout(fetchPrediction, 120);
}

async function fetchPrediction() {
  const id = ++state.requestId;
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(state.values),
    });
    if (!response.ok) throw new Error(`Prediction failed (${response.status})`);
    const data = await response.json();
    if (id === state.requestId) renderPrediction(data.prediction);
  } catch (error) {
    console.error(error);
    $("result-note").textContent = "The estimate could not be updated. Check the server logs.";
  }
}

function bindEvents() {
  $("country-preset").addEventListener("change", (event) => {
    const name = event.target.value;
    if (!name) return;
    const country = state.model.countries.find((c) => c.name === name);
    if (!country) return;
    state.values = { ...country.values };
    syncInputs(name);
    schedulePrediction();
  });

  $("reset-btn").addEventListener("click", () => {
    state.values = { ...state.model.defaults };
    syncInputs();
    schedulePrediction();
  });

  $("country-search").addEventListener("input", (event) => {
    const query = event.target.value.trim().toLowerCase();
    let visible = 0;
    for (const row of document.querySelectorAll("#country-body tr")) {
      const match = row.dataset.name.includes(query);
      row.hidden = !match;
      if (match) visible += 1;
    }
    $("table-meta").textContent = query
      ? `${visible} of ${state.model.countries.length} countries`
      : `${state.model.countries.length} countries · predicted values from the fitted regression`;
  });
}

async function init() {
  try {
    const response = await fetch("/api/model");
    if (!response.ok) throw new Error(`Model request failed (${response.status})`);
    state.model = await response.json();
  } catch (error) {
    console.error(error);
    $("result-note").textContent = "The model could not be loaded. Check the server logs.";
    return;
  }

  state.values = { ...state.model.defaults };
  renderStats();
  renderPresets();
  renderControls();
  renderCoefficients();
  renderTable();
  bindEvents();
  syncInputs();
  await fetchPrediction();
}

init();

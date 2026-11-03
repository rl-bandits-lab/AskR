(() => {
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false },
      ],
    });
  }

  const afterLayout = (fn) => requestAnimationFrame(() => requestAnimationFrame(fn));

  document.querySelectorAll(".player").forEach((player) => {
    const last = Number(player.dataset.steps) - 1;
    const railButtons = [...player.querySelectorAll(".rail button")];
    const captions = [...player.querySelector("template.captions").content.children];
    const caption = player.querySelector(".caption");
    const prev = player.querySelector(".prev");
    const next = player.querySelector(".next");
    const play = player.querySelector(".play");
    let step = 0;
    let timer = null;

    function render() {
      player.querySelectorAll("[data-from]").forEach((el) => {
        const from = Number(el.dataset.from);
        const to = el.dataset.to ? Number(el.dataset.to) : Infinity;
        const visible = step >= from && step <= to;
        if (!el.closest(".transcript")) return el.classList.toggle("on", visible);
        if (!visible) return el.classList.remove("on", "shown");
        if (el.classList.contains("on")) return;
        el.classList.add("shown");
        afterLayout(() => el.classList.add("on"));
      });
      player.querySelectorAll("[data-answered]").forEach((el) => el.classList.toggle("answered", step >= Number(el.dataset.answered)));
      player.querySelectorAll("[data-final]").forEach((el) => el.classList.toggle("on-final", step >= Number(el.dataset.final)));

      railButtons.forEach((b, i) => {
        b.classList.toggle("done", i < step);
        if (i === step) b.setAttribute("aria-current", "step");
        else b.removeAttribute("aria-current");
      });
      caption.innerHTML = captions[step].innerHTML;
      prev.disabled = step === 0;
      next.disabled = step === last;

    }

    function go(s) {
      step = Math.max(0, Math.min(last, s));
      render();
    }

    function setPlaying(on) {
      play.innerHTML = on
        ? '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg>'
        : '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
      play.setAttribute("aria-label", on ? "Pause" : "Play");
    }

    function pause() {
      clearInterval(timer);
      timer = null;
      setPlaying(false);
    }

    function start() {
      if (step === last) go(0);
      setPlaying(true);
      timer = setInterval(() => {
        if (step >= last) return pause();
        go(step + 1);
      }, 3400);
    }

    prev.addEventListener("click", () => { pause(); go(step - 1); });
    next.addEventListener("click", () => { pause(); go(step + 1); });
    play.addEventListener("click", () => (timer ? pause() : start()));
    railButtons.forEach((b, i) => b.addEventListener("click", () => { pause(); go(i); }));
    player.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight") { pause(); go(step + 1); }
      if (e.key === "ArrowLeft") { pause(); go(step - 1); }
    });

    render();
  });

  const signer = document.getElementById("signer");
  if (signer) {
    const slider = signer.querySelector("#r-slider");
    const out = signer.querySelector("#r-out");
    const bar = signer.querySelector(".meter .bar");
    const verdict = signer.querySelector(".verdict");
    const segButtons = [...signer.querySelectorAll(".seg button")];
    let correct = true;

    function update() {
      const r = Number(slider.value);
      const R = correct ? r : -r;
      out.textContent = r.toFixed(2);
      const half = Math.abs(R) * 50;
      bar.style.left = `${R >= 0 ? 50 : 50 - half}%`;
      bar.style.width = `${half}%`;
      bar.style.background = R >= 0 ? "var(--sage)" : "var(--signal)";

      let headline, detail;
      if (correct && r >= 0.5) {
        headline = `R = +${r.toFixed(2)}: reinforced`;
        detail = "A right answer with high self-verification. This is the pattern AskR should produce.";
      } else if (correct) {
        headline = `R = +${r.toFixed(2)}: weakly reinforced`;
        detail = "Right answer, but the checks did not vouch for it. Questions that make the checks agree earn more.";
      } else if (r >= 0.5) {
        headline = `R = −${r.toFixed(2)}: penalized`;
        detail = "A wrong answer that looks verified. These questions make the reward misleading, so they are pushed down.";
      } else {
        headline = `R = −${r.toFixed(2)}: barely penalized`;
        detail = "A wrong answer the checks already flagged. The reward was honest, so the penalty is small.";
      }
      verdict.innerHTML = `${headline}<small>${detail}</small>`;
    }

    segButtons.forEach((b) => b.addEventListener("click", () => {
      correct = b.dataset.correct === "1";
      segButtons.forEach((x) => x.setAttribute("aria-pressed", x === b));
      update();
    }));
    slider.addEventListener("input", update);
    update();
  }

  const COLS = ["mmlu", "lab", "sci", "langAvg", "vstar", "mmstar", "cv", "visAvg"];
  const MAIN = [
    ["Training-based reference"],
    ["RLVR", [64.19, 33.75, 68.78, 55.57, 79.50, 60.93, 72.91, 71.11]],
    ["Prompting-based"],
    ["CoT", [55.75, 32.63, 48.75, 45.71, 72.50, 59.13, 68.47, 66.70]],
    ["Least-to-Most", [56.70, 29.80, 43.23, 43.24, 72.94, 55.21, 68.96, 65.70]],
    ["Tree of Thoughts", [54.00, 33.72, 43.27, 43.66, 75.06, 60.53, 66.25, 67.28]],
    ["Self-Refine", [57.21, 33.53, 59.53, 50.09, 73.53, 57.00, 66.34, 65.62]],
    ["Value-model guided"],
    ["VM + VGS", [55.36, 32.20, 48.70, 45.42, 71.24, 61.83, 74.60, 69.22]],
    ["VM + PF", [61.03, 31.89, 51.99, 48.30, 78.14, 61.05, 71.74, 70.31]],
    ["Best-of-N"],
    ["PRM + BoN", [58.40, 30.49, 49.67, 46.19, 72.75, 58.80, 69.54, 67.03]],
    ["AskR + BoN", [61.23, 35.05, 65.43, 53.90, 77.48, 61.73, 72.53, 70.58], true],
    ["TPO"],
    ["PRM + TPO", [56.18, 29.70, 48.50, 44.79, 78.01, 42.66, 72.46, 64.38]],
    ["AskR + TPO", [60.61, 33.19, 63.10, 52.30, 76.43, 62.40, 71.07, 69.97], true],
    ["TMPC"],
    ["PRM + TMPC", [60.30, 31.17, 47.86, 46.44, 75.78, 58.40, 66.98, 67.05]],
    ["AskR + TMPC", [63.76, 37.29, 70.45, 57.17, 84.37, 63.23, 76.43, 74.68], true],
  ];
  const FAMILY = [
    ["Qwen3-4B"],
    ["CoT", [69.60, 34.98, 71.60, 58.73, 82.72, 66.00, 58.50, 69.07]],
    ["RLVR", [73.98, 36.50, 76.15, 62.21, 80.10, 68.50, 59.85, 69.48]],
    ["AskR + TMPC", [78.42, 44.83, 76.46, 66.57, 84.50, 70.33, 66.00, 73.61], true],
    ["Gemma3-4B"],
    ["CoT", [41.24, 26.50, 58.50, 42.08, 36.65, 52.50, 27.50, 38.88]],
    ["RLVR", [45.92, 26.00, 63.50, 45.14, 37.17, 54.50, 26.00, 39.22]],
    ["AskR + TMPC", [47.45, 32.73, 64.62, 48.27, 46.21, 58.31, 34.50, 46.34], true],
  ];

  function fillTable(tbody, rows) {
    if (!tbody) return;
    tbody.innerHTML = rows.map(([name, vals, askr]) => {
      if (!vals) return `<tr class="group"><th colspan="5" scope="rowgroup">${name}</th><td colspan="4" class="split"></td></tr>`;
      const cells = vals.map((v, i) => {
        const cls = [i === 3 || i === 7 ? "avg" : "", i === 4 ? "split" : ""].join(" ").trim();
        return `<td${cls ? ` class="${cls}"` : ""}>${v.toFixed(2)}</td>`;
      }).join("");
      return `<tr${askr ? ' class="askr"' : ""}><th scope="row">${name}</th>${cells}</tr>`;
    }).join("");
  }
  fillTable(document.getElementById("full-table"), MAIN);
  fillTable(document.getElementById("family-table"), FAMILY);

  const svg = document.getElementById("bars");
  if (svg) {
    const NS = "http://www.w3.org/2000/svg";
    const lookup = Object.fromEntries(MAIN.filter((r) => r[1]).map(([n, v]) => [n, v]));
    const OPT = ["BoN", "TPO", "TMPC"];
    const W = 720, H = 340, L = 40, R = 104, T = 30, B = 56;
    const plotW = W - L - R, plotH = H - T - B;
    const groupW = plotW / OPT.length, barW = 58, gap = 10;

    const el = (tag, attrs, parent = svg) => {
      const n = document.createElementNS(NS, tag);
      Object.entries(attrs).forEach(([k, v]) => n.setAttribute(k, v));
      parent.appendChild(n);
      return n;
    };

    const grid = el("g", {});
    const refs = ["RLVR", "CoT"].map((name) => ({
      name,
      line: el("line", { x1: L, x2: L + plotW, stroke: name === "RLVR" ? "#bf3f37" : "#7c8496", "stroke-width": 2, "stroke-dasharray": "6 5" }),
      label: el("text", { x: L + plotW + 8, "font-size": 13, fill: name === "RLVR" ? "#bf3f37" : "#7c8496", "dominant-baseline": "middle" }),
    }));
    const groups = OPT.map((opt, g) => {
      const cx = L + groupW * g + groupW / 2;
      const make = (name, x, fill) => ({
        name,
        rect: el("rect", { x, width: barW, rx: 6, fill }),
        val: el("text", { x: x + barW / 2, "text-anchor": "middle", "font-size": 14, "font-weight": 600, fill: "#1c2230", class: "val" }),
      });
      const prm = make(`PRM + ${opt}`, cx - barW - gap / 2, "#a3abbb");
      const askr = make(`AskR + ${opt}`, cx + gap / 2, "#6a4c9c");
      el("text", { x: cx, y: H - B + 24, "text-anchor": "middle", "font-size": 15, "font-weight": 700, fill: "#1c2230" }).textContent = opt;
      const delta = el("text", { x: cx, y: H - B + 44, "text-anchor": "middle", "font-size": 14, "font-weight": 700, fill: "#2c7a52" });
      return { prm, askr, delta };
    });
    el("line", { x1: L, x2: L + plotW, y1: T + plotH, y2: T + plotH, stroke: "#1c2230", "stroke-width": 1.5 });

    let current = null;
    function draw(key, animate) {
      const idx = COLS.indexOf(key);
      const values = [...groups.flatMap((g) => [lookup[g.prm.name][idx], lookup[g.askr.name][idx]]),
        lookup.RLVR[idx], lookup.CoT[idx]];
      const lo = Math.max(0, Math.floor((Math.min(...values) - 6) / 5) * 5);
      const hi = Math.ceil((Math.max(...values) + 3) / 5) * 5;
      const y = (v) => T + plotH - ((v - lo) / (hi - lo)) * plotH;

      grid.innerHTML = "";
      const stepTick = hi - lo > 30 ? 10 : 5;
      for (let v = lo; v <= hi; v += stepTick) {
        el("line", { x1: L, x2: L + plotW, y1: y(v), y2: y(v), stroke: "#e6e9f0" }, grid);
        el("text", { x: L - 8, y: y(v), "text-anchor": "end", "dominant-baseline": "middle", "font-size": 12, fill: "#7c8496" }, grid).textContent = v;
      }

      const targets = [];
      groups.forEach((g) => {
        [g.prm, g.askr].forEach((b) => targets.push([b, lookup[b.name][idx]]));
        const d = lookup[g.askr.name][idx] - lookup[g.prm.name][idx];
        g.delta.textContent = `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(2)}`;
        g.delta.setAttribute("fill", d >= 0 ? "#2c7a52" : "#bf3f37");
      });

      const from = current || Object.fromEntries(targets.map(([b]) => [b.name, lo]));
      const fromRefs = current ? current._refs : {};
      const duration = animate && !reduceMotion ? 520 : 0;
      const t0 = performance.now();
      function frame(now) {
        const p = duration ? Math.min(1, (now - t0) / duration) : 1;
        const e = 1 - Math.pow(1 - p, 3);
        targets.forEach(([b, v]) => {
          const cur = from[b.name] + (v - from[b.name]) * e;
          const top = y(Math.max(cur, lo));
          b.rect.setAttribute("y", top);
          b.rect.setAttribute("height", T + plotH - top);
          b.val.setAttribute("y", top - 8);
          b.val.textContent = v.toFixed(2);
        });
        refs.forEach((r) => {
          const v = lookup[r.name][idx];
          const start = fromRefs[r.name] ?? v;
          const yy = y(start + (v - start) * e);
          r.line.setAttribute("y1", yy);
          r.line.setAttribute("y2", yy);
          r.label.setAttribute("y", yy);
          r.label.textContent = `${r.name} ${v.toFixed(2)}`;
        });
        if (p < 1) requestAnimationFrame(frame);
      }
      requestAnimationFrame(frame);
      current = Object.fromEntries(targets.map(([b, v]) => [b.name, v]));
      current._refs = Object.fromEntries(refs.map((r) => [r.name, lookup[r.name][idx]]));
    }

    const chips = [...document.querySelectorAll(".chips button")];
    chips.forEach((c) => c.addEventListener("click", () => {
      chips.forEach((x) => x.setAttribute("aria-pressed", x === c));
      draw(c.dataset.key, true);
    }));
    draw("langAvg", false);
  }

  const viewButtons = [...document.querySelectorAll(".view-toggle button")];
  viewButtons.forEach((b) => b.addEventListener("click", () => {
    viewButtons.forEach((x) => x.setAttribute("aria-pressed", x === b));
    document.querySelectorAll(".chart-card .view").forEach((v) => { v.hidden = v.dataset.view !== b.dataset.view; });
  }));

  const copy = document.getElementById("copy-bib");
  if (copy) {
    copy.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(document.getElementById("bibtex").textContent);
        copy.textContent = "Copied";
      } catch {
        copy.textContent = "Select and copy";
      }
      setTimeout(() => { copy.textContent = "Copy"; }, 1800);
    });
  }
})();

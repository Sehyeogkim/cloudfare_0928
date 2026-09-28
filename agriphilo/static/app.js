// Shared client for the WEMINE console: API calls and formatting helpers.
window.Agri = (() => {
  async function api(method, path, body) {
    const r = await fetch(path, {
      method,
      headers: { "Content-Type": "application/json" },
      body: body ? JSON.stringify(body) : undefined,
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.error || r.statusText);
    return data;
  }

  // Coverage axes (integer ids or stage counts), kept for the QA views.
  const AXIS_FMT = {
    layout_id: { label: "Counter layouts", f: (v) => `layout ${v}` },
    style_id: { label: "Visual styles", f: (v) => `style ${v}` },
    stages_completed: { label: "Task stages completed", f: (v) => `${v} stages` },
  };

  const SENSOR_LABELS = {
    rgb: "RGB", depth: "Depth", joint_state: "Joint state", eef_pose: "End-effector pose",
    gripper_state: "Gripper state", action: "Action", object_pose: "Object pose", task_stage: "Task stage",
  };

  const CAMERA_LABELS = {
    robot0_agentview_center: "Agent view (center)", robot0_agentview_left: "Agent view (left)",
    robot0_agentview_right: "Agent view (right)", robot0_frontview: "Front view", robot0_eye_in_hand: "Wrist camera",
  };

  function successText(sr) {
    return `All task stages completed within ${sr.time_limit_s} s`;
  }

  const shortId = (id) => "ORD-" + String(id).slice(0, 6).toUpperCase();

  function title(text) {
    const line = String(text || "").trim().split("\n")[0];
    return line.length > 90 ? line.slice(0, 88) + "…" : line;
  }

  function ago(ts) {
    const s = Math.max(0, Math.round(Date.now() / 1000 - ts));
    if (s < 60) return `${s}s ago`;
    if (s < 3600) return `${Math.round(s / 60)}m ago`;
    if (s < 86400) return `${Math.round(s / 3600)}h ago`;
    return `${Math.round(s / 86400)}d ago`;
  }

  function elapsed(from, to) {
    const s = Math.max(0, Math.round((to || Date.now() / 1000) - from));
    return s < 60 ? `${s}s` : s < 3600 ? `${Math.floor(s / 60)}m ${s % 60}s` : `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  }

  const clock = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const usd = (v) => (v < 0 ? "−$" : "$") + Math.abs(v).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  function bytes(n) {
    if (n == null) return "";
    if (n < 1024) return `${n} B`;
    if (n < 1 << 20) return `${(n / 1024).toFixed(1)} KB`;
    if (n < 1 << 30) return `${(n / (1 << 20)).toFixed(1)} MB`;
    return `${(n / (1 << 30)).toFixed(2)} GB`;
  }

  // Tiny element builder: h("div.card", {onclick}, child, "text", ...)
  function h(tag, attrs, ...children) {
    const [name, ...classes] = tag.split(".");
    const el = document.createElement(name || "div");
    if (classes.length) el.className = classes.join(" ");
    if (attrs && (typeof attrs !== "object" || attrs instanceof Node || Array.isArray(attrs))) {
      children.unshift(attrs);
    } else if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
        else if (k === "html") el.innerHTML = v; // static markup only, never server data
        else if (v !== false && v != null) el.setAttribute(k, v === true ? "" : v);
      }
    }
    for (const c of children.flat(Infinity)) {
      if (c == null || c === false) continue;
      el.append(c instanceof Node ? c : document.createTextNode(String(c)));
    }
    return el;
  }

  return {
    config: () => api("GET", "/api/config"),
    examples: () => api("GET", "/api/examples"),
    list: () => api("GET", "/api/orders").then((d) => d.items),
    get: (id) => api("GET", `/api/orders/${id}`),
    create: (text, auto) => api("POST", "/api/orders", { text, auto }),
    answer: (id, text) => api("POST", `/api/orders/${id}/answer`, { text }),
    approve: (id) => api("POST", `/api/orders/${id}/approve`),
    purchase: (id, count) => api("POST", `/api/orders/${id}/purchase`, { count }),
    close: (id) => api("POST", `/api/orders/${id}/close`),
    reopen: (id) => api("POST", `/api/orders/${id}/reopen`),
    wallet: () => api("GET", "/api/wallet"),
    walletCheckout: (credits) => api("POST", "/api/wallet/checkout", { credits }),
    walletConfirm: (sessionId) => api("POST", "/api/wallet/confirm", { session_id: sessionId }),
    walletTestTopup: (amount) => api("POST", "/api/wallet/topup", { amount }),
    retry: (id) => api("POST", `/api/orders/${id}/retry`),
    fileUrl: (id, name) => `/api/orders/${id}/files/${name}`,
    // Players: id is derived from the signed-in email (the server uses the same rule).
    playerId: (email) => String(email || "").trim().toLowerCase().replace(/[^a-z0-9._-]/g, "-"),
    marketplace: () => api("GET", "/api/marketplace").then((d) => d.items),
    player: (id) => api("GET", `/api/players/${encodeURIComponent(id)}`),
    successText, shortId, title, ago, elapsed, clock, usd, bytes, h,
    SENSOR_LABELS, CAMERA_LABELS, AXIS_FMT,
  };
})();

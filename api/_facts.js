// Shared, always-current facts for the page (/api/stats) and the chatbot (/api/chat).
// Static facts live in data/profile.json; GitHub numbers are fetched live and cached.
// Optional env var: GITHUB_TOKEN (any read-only token) raises GitHub's search rate limit.

const profile = require('../data/profile.json');

const TTL_MS = 6 * 60 * 60 * 1000;      // refresh live numbers every 6 hours
const RETRY_MS = 10 * 60 * 1000;        // after a failure, try again in 10 minutes
let cache = null, cacheAt = 0, inflight = null;

async function ghSearch(q) {
  const headers = { 'User-Agent': 'ashish-portfolio', Accept: 'application/vnd.github+json' };
  if (process.env.GITHUB_TOKEN) headers.Authorization = `Bearer ${process.env.GITHUB_TOKEN}`;
  const r = await fetch(`https://api.github.com/search/issues?per_page=100&q=${encodeURIComponent(q)}`, { headers });
  if (!r.ok) throw new Error(`github search ${r.status}`);
  return r.json();
}

async function fetchLive() {
  const u = profile.github.user, spot = profile.github.spotlightRepo;
  const [merged, open] = await Promise.all([
    ghSearch(`author:${u} type:pr is:merged -user:${u}`),
    ghSearch(`author:${u} type:pr is:open -user:${u}`),
  ]);
  const byRepo = {};
  (merged.items || []).forEach(i => {
    const repo = i.repository_url.split('/repos/')[1];
    byRepo[repo] = (byRepo[repo] || 0) + 1;
  });
  let contributions = profile.github.fallback.contributionsLastYear;
  try {
    const c = await (await fetch(`https://github-contributions-api.jogruber.de/v4/${u}?y=last`)).json();
    if (c && c.total && typeof c.total.lastYear === 'number') contributions = c.total.lastYear;
  } catch (e) {}
  return {
    mergedPRs: merged.total_count,
    tensormapPRs: byRepo[spot] || 0,
    otherProjects: Object.keys(byRepo).filter(r => r !== spot).length,
    openPRs: open.total_count,
    mergedByRepo: byRepo,
    contributionsLastYear: contributions,
    asOf: new Date().toISOString(),
    source: 'github',
  };
}

async function liveStats() {
  const fresh = cache && Date.now() - cacheAt < TTL_MS;
  if (fresh) return cache;
  if (!inflight) {
    inflight = fetchLive()
      .then(v => { cache = v; cacheAt = Date.now(); return v; })
      .catch(() => {
        if (!cache) cache = Object.assign({ source: 'fallback' }, profile.github.fallback);
        cacheAt = Date.now() - TTL_MS + RETRY_MS;
        return cache;
      })
      .finally(() => { inflight = null; });
  }
  return inflight;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
function monthLabel(iso) { const [y, m] = iso.split('-'); return { y, m: MONTHS[+m - 1] }; }

// Status is derived from the dates, so "Now" flips to "Completed" on its own.
function internships(now = new Date()) {
  return profile.internships.map(r => {
    const ended = new Date(`${r.end}T23:59:59+05:30`) < now;
    const s = monthLabel(r.start), e = monthLabel(r.end);
    const when = ended
      ? (s.y === e.y ? `${s.m} — ${e.m} ${e.y}` : `${s.m} ${s.y} — ${e.m} ${e.y}`)
      : `${s.m} ${s.y} — Now`;
    return Object.assign({}, r, { status: ended ? 'completed' : 'current', when });
  });
}

async function getFacts() {
  // Never let a slow GitHub call hold up a chat answer or page load.
  const timeout = new Promise(res => setTimeout(() => res(cache || Object.assign({ source: 'fallback' }, profile.github.fallback)), 3500));
  const live = await Promise.race([liveStats(), timeout]);
  const roles = internships();
  return {
    profile,
    live,
    internships: roles,
    internshipCount: roles.length,
    currentRole: roles.find(r => r.status === 'current') || null,
    generatedAt: new Date().toISOString(),
  };
}

// The same facts as plain text, for the chatbot's context.
function factsText(f) {
  const p = f.profile, l = f.live, ed = p.education;
  const search = `https://github.com/search?q=author%3A${p.github.user}+type%3Apr+is%3Amerged+-user%3A${p.github.user}&type=pullrequests`;
  const spot = `https://github.com/${p.github.spotlightRepo}/pulls?q=is%3Apr+is%3Amerged+author%3A${p.github.user}`;
  const roles = f.internships.map(r => `- ${r.company}: ${r.role}, ${r.when} (${r.status}). ${r.summary}`).join('\n');
  return [
    `## Live facts (as of ${f.generatedAt.slice(0, 10)})`,
    `These are the authoritative, current numbers. Prefer them over any other figure.`,
    `CGPA: ${ed.cgpa} / ${ed.cgpaScale} (${ed.degree}, ${ed.school}, ${ed.start} to ${ed.end} expected).`,
    `Internships completed or in progress: ${f.internshipCount}.`,
    roles,
    f.currentRole
      ? `Current role: ${f.currentRole.role} at ${f.currentRole.company}.`
      : `Current status: not in an internship right now. ${p.availability.headline}. ${p.availability.detail}.`,
    `Open source (live from GitHub${l.source === 'fallback' ? ', cached' : ''}): ${l.mergedPRs} merged pull requests to other people's projects, ${l.tensormapPRs} of them in ${p.github.spotlightRepo}, plus ${l.otherProjects} other projects; ${l.openPRs} more open or in review.`,
    `Evidence: [all merged PRs](${search}) · [TensorMap PRs](${spot})`,
    `GitHub contributions in the last year: ${l.contributionsLastYear}.`,
  ].join('\n');
}

module.exports = { getFacts, factsText };

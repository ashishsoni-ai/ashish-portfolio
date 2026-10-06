// GET /api/stats — current facts for the page: CGPA, internships (status from dates),
// availability, and live GitHub numbers. Cached at Vercel's edge so GitHub is hit rarely.
const { getFacts } = require('./_facts.js');

module.exports = async function handler(req, res) {
  const f = await getFacts();
  res.setHeader('Cache-Control', 'public, max-age=300, s-maxage=21600, stale-while-revalidate=86400');
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.status(200).json({
    cgpa: f.profile.education.cgpa,
    cgpaScale: f.profile.education.cgpaScale,
    availability: f.profile.availability,
    internshipCount: f.internshipCount,
    internships: f.internships.map(r => ({ id: r.id, company: r.company, role: r.role, when: r.when, status: r.status })),
    currentRole: f.currentRole ? { company: f.currentRole.company, role: f.currentRole.role } : null,
    github: f.live,
    generatedAt: f.generatedAt,
  });
};

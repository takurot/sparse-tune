# Security policy

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use the repository's
private vulnerability reporting channel. Include the affected version, a
minimal reproducer, impact, and any suggested mitigation.

## Continuous controls

- Dependabot checks Python and GitHub Actions dependencies monthly. Security
  updates and vulnerability alerts are enabled in the repository settings.
- The Security workflow audits resolved Python dependencies with `pip-audit`
  and uploads CodeQL analysis for pull requests, `main`, and a weekly schedule.
- Every referenced GitHub Action is pinned to a reviewed commit SHA. The
  adjacent version comment and Dependabot update preserve reviewable upgrades.
- The default-branch ruleset requires every Test workflow job before merge.
  Workflow permissions remain read-only unless a job needs the narrowly scoped
  `security-events: write` or OIDC `id-token: write` permission.

## Finding triage

A security scan failure is actionable until the finding is reproduced and its
data and process boundaries are reviewed. Severity, exploitability, affected
versions, and remediation or acceptance rationale must be recorded in the
pull request or a dedicated issue.

The project intentionally launches isolated Python worker modules and the
fixed `sysctl` executable with argument arrays, `shell=False`, bounded
timeouts, and validated result files. Generic warnings about importing
`subprocess` or invoking a process without a shell (for example Bandit B404,
B603, and B607) are accepted only for those reviewed call sites. A command
constructed from untrusted input, use of `shell=True`, or an unbounded process
is not covered by this exception.

## Solo-maintainer emergency procedure

The ruleset has no standing bypass. If GitHub Actions is unavailable or a
required check is irreparably misconfigured, the repository administrator may
temporarily disable only the affected required check, record the reason and
before/after ruleset state in an issue, merge the smallest recovery change,
and immediately restore the rule. Deletion and non-fast-forward protections
remain enabled throughout.

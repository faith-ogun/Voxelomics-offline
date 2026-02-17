# GitHub Release Checklist (Hackathon Ready)

Use this checklist for every release candidate sent to judges.

## A. Pre-release sanity

- [ ] Run from clean repo state: `git status` is clean.
- [ ] Frontend builds: `npm run build`
- [ ] Desktop package builds: `npm run desktop:dist`
- [ ] Verify app launches locally from packaged output.
- [ ] Run one full demo flow:
  - [ ] `Load Demo Dictation`
  - [ ] `Run MDT Pipeline`
  - [ ] Check `DiagnostiCore` WSI controls render correctly
  - [ ] Confirm `Patient Cases` snapshot load works

## B. Verify build artifacts

- [ ] Confirm output files exist under `dist/` (or configured output dir).
- [ ] Confirm file names are clear and platform-specific.
- [ ] Confirm no private or irrelevant files are bundled.

## C. GitHub release publishing

- [ ] Push final commit to `main`.
- [ ] Create tag (example): `v0.1.0-hackathon`.
- [ ] Create GitHub Release from that tag.
- [ ] Upload installer/zip assets.
- [ ] Paste release notes from `RELEASE_NOTES_TEMPLATE.md`.
- [ ] Include direct link to `JUDGE_QUICKSTART.md`.

## D. Repo hygiene check

- [ ] README has:
  - [ ] setup steps
  - [ ] desktop packaging steps
  - [ ] judge quick start link
- [ ] `.gitignore` excludes generated/runtime files.
- [ ] No stray draft docs or unrelated legacy scripts.
- [ ] Required licenses and attributions are present.

## E. Submission bundle alignment

- [ ] Video is <= 3 minutes.
- [ ] Write-up <= 3 pages and follows template.
- [ ] Public code repo link works.
- [ ] Optional live demo note is accurate (desktop-first offline demo).
- [ ] Track selection is set (Main + Agentic Workflow prize).

## F. Last 15-minute lock

- [ ] Re-download release asset from GitHub and test once.
- [ ] Confirm all links in release notes work.
- [ ] Freeze commit hash used in write-up/video.
- [ ] Submit Kaggle write-up package.

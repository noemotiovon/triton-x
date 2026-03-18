---
applyTo: "src/**/manifest.yaml,config/**"
---

# Project Conventions

When editing manifest or config files, read the conventions skill first:

**Skill file**: `.ai/skills/project-conventions.md`

Key rules:
- Follow manifest status transitions: pending → converted → testing → verified/failed
- Never hardcode accuracy thresholds — use `config/accuracy.yaml`
- Always update `manifest.yaml` after kernel operations

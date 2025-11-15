# report.py
from jinja2 import Template
import os

REPORT_TEMPLATE = """
# Multi-Agent Analysis Report

## Executive summary
{{ insight }}

## Evidence
{% for key, value in evidence.items() %}
### {{ key }}
{{ value }}
{% endfor %}

## Visuals
{% for img in images %}
- ![]({{ img }})
{% endfor %}
"""

def compose_report(insight_text, evidence_dict, images, out_path="results/report.md"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tpl = Template(REPORT_TEMPLATE)
    rendered = tpl.render(insight=insight_text, evidence=evidence_dict, images=images)
    with open(out_path, "w") as f:
        f.write(rendered)
    return out_path

# Resume Matching Engine - Data Specification

> **Version**: 1.0  
> **Status**: DRAFT - Pending Review  
> **Last Updated**: 2026-01-19

---

## 1. Overview

This document specifies the synthetic evaluation dataset, including:
- Job description(s) for testing
- Sample resumes with ground truth labels
- Labeling criteria and methodology

---

## 2. Job Description: AI Applications Engineer @ Ema

We are using the actual Ema job posting as our primary matching target.

```
============================================================
AI APPLICATIONS ENGINEER @ EMA
Location: Bengaluru, India (or Remote)
============================================================

WHO ARE WE?
-----------
Ema is building the next generation AI technology to empower 
every employee in the enterprise. Our proprietary tech allows 
enterprises to delegate repetitive tasks to Ema, the Universal 
AI employee.

YOU WILL:
---------
• Configure and integrate AI/GenAI workflows using platform 
  APIs and customer data.
• Troubleshoot and resolve production issues related to platform 
  APIs, data integration, and workflow configurations.
• Design, execute, and analyze A/A experiments to enhance 
  workflow reliability and quality.
• Monitor workflow performance using observability dashboards 
  and business metrics.
• Refine prompts for optimized performance.
• Translate customer requirements into effective GenAI 
  automation flows.

IDEALLY, YOU'D HAVE:
--------------------
• 3+ years in technical support or customer success for SaaS.
• Proficiency in troubleshooting production issues and 
  diagnosing ML performance.
• Familiarity with APIs (JSON, REST, SOAP) and integration with 
  CRMs or ATS tools.
• Competency in logging, dashboard creation, and alerting tools.
• Strong proficiency in Python.
============================================================
```

### 2.2 Key Skills Extracted

| Category | Required Skills | Nice-to-Have |
|----------|-----------------|--------------|
| **AI/GenAI** | Prompt Engineering, GenAI Workflows | RAG, Agentic Systems |
| **Engineering** | Python, API (REST/JSON) | SaaS Support, CRM Integration |
| **Operations** | Observability, Dashboards | A/B Testing, ML Diagnostics |
| **Infrastructure** | API Integration | AWS, Terraform |

---

## 3. Labeling Rubric (The AI Engineering Logic)

To evaluate the resumes objectively, we use a 3-category rubric based on the **Ema AI Applications Engineer** requirements.

| Category | Weight | Ideal Indicators |
|----------|--------|------------------|
| **GenAI/LLM** | 40% | Prompt engineering, RAG, AI agents, model performance troubleshooting. |
| **Tech Support** | 30% | 3+ years in B2B SaaS/Enterprise support, customer-facing troubleshooting. |
| **Tech Proficiency**| 30% | Python (backend/API focus), REST APIs, SQL, Datadog/monitoring. |

### Score Determination
*   **Good Match (1.0)**: Strong in **GenAI** + **Enterprise Support** + **Technical proficiency**.
*   **Partial Match (0.5)**: Strong in **Support/Technical** but lacks **GenAI**. OR strong in **GenAI/Technical** but lacks **SaaS Support** experience.
*   **Poor Match (0.0)**: Lacks relevance in 2 or more core categories. Wrong tech stack (e.g. Pure Frontend, No Python).

---

## 4. Resume Dataset (10 Profiles)

### 4.1 Dataset Summary

| Resume ID | Profile Summary | Expected Label | Rationale |
|-----------|-----------------|----------------|-----------|
| res_001 | **Alex Chen** (AI Support Eng) | **Good (1.0)** | ML workflows + SaaS Support + Python expert. |
| res_002 | **Sarah Johnson** (Sr. Support Eng) | **Good (1.0)** | Heavy Enterprise Support + AWS/Python. High performance. |
| res_003 | **Maya Gupta** (Solutions Eng) | **Good (1.0)** | AI integration expert + Customer facing. GenAI focus. |
| res_004 | **David Kim** (Backend/APIs) | **Partial (0.5)** | Strong API dev, but missing direct GenAI/Prompting experience. |
| res_005 | **Mike Rodriguez** (Full-Stack) | **Partial (0.5)** | Has Python/SaaS, but focus is UI features, not Support or AI performance. |
| res_006 | **Priya Sharma** (Data Scientist) | **Partial (0.5)** | Strong AI/Python, but no Customer Support/SaaS experience. |
| res_007 | **Sam Taylor** (Cust. Success) | **Partial (0.5)** | Deep SaaS support background, but limited technical/coding depth. |
| res_008 | **Jordan Lee** (Frontend Dev) | **Poor (0.0)** | UI/UX focus only. No match for AI application support. |
| res_009 | **James Wilson** (Java Backend) | **Poor (0.0)** | Strong eng, but wrong stack (Java) and no AI/Support domain experience. |
| res_010 | **Elena Rossi** (Marketing) | **Poor (0.0)** | No relevant technical or customer engineering background. |

---

### 4.2 Resume Profiles (Concise)

#### res_001: Yashpreet Voladoddi (Good - 1.0)
*Profile: SDE & AI Engineer (Kapi AI / HeyCoach)*
- **JD Fit**: Excellent technical alignment with Agentic workflows (LangGraph/LangChain) and prompt engineering. Strong Python/AWS background. 
- **Experience Gap**: ~2 years of professional/intern experience; slightly below the "3+ years" ideal support tenure, but technical depth in Ema's core domain (AI Employees) makes him highly relevant.

#### res_002: Sarah Johnson (Partial - 0.5)
*Profile: Senior Enterprise Support Engineer*
- **JD Fit**: Meets and exceeds the support years (7+ yrs) and Python requirements. 
- **Experience Gap**: No direct experience with LLMs, prompt engineering, or RAG. Per the rubric, missing the core GenAI category makes this a partial match for this specific AI-native role.

#### res_003: Maya Gupta (Good - 1.0)
*Profile: AI Solutions Engineer*
- **JD Fit**: 4 years experience. Built customer-facing GenAI integrations. 
- **Experience**: Direct overlap with prompt engineering and translating customer requirements into AI flows.

#### res_004: David Kim (Partial - 0.5)
*Profile: Backend / Integration Engineer*
- **JD Fit**: Strong API troubleshooting (Go/Python), missing direct GenAI.
- **Experience**: Expert in REST integrations and data sync issues.

#### res_005: Mike Rodriguez (Partial - 0.5)
*Profile: Full-Stack Web Developer*
- **JD Fit**: Feature development focus. Strong Python/Flask.
- **Experience**: Builds consumer apps; lacks enterprise support/observability depth.

#### res_006: Priya Sharma (Partial - 0.5)
*Profile: Data Scientist*
- **JD Fit**: High AI depth, but no application-level support experience.
- **Experience**: Predictive modeling and data pipelines; not an App Engineer.

#### res_007: Sam Taylor (Partial - 0.5)
*Profile: Technical Customer Success Manager*
- **JD Fit**: Deep SaaS support workflows, but limited coding depth.
- **Experience**: Managed technical escalations; strong SQL but low Python proficiency.

#### res_008: Jordan Lee (Poor - 0.0)
*Profile: Frontend Developer*
- **JD Fit**: UI focus only. No Python, no Backend, no Support.
- **Experience**: React/Vue specialist.

#### res_009: James Wilson (Poor - 0.0)
*Profile: Enterprise Java Developer*
- **JD Fit**: Wrong stack (Java). No AI or Support domain experience.
- **Experience**: Built high-volume banking backends in Spring Boot.

#### res_010: Elena Rossi (Poor - 0.0)
*Profile: Marketing Manager*
- **JD Fit**: No technical or customer engineering skills.
- **Experience**: SEO and content strategy.

---

## 5. Ground Truth Labels (evaluation_set.json)

```json
{
  "job_description_id": "ema_ai_app_eng",
  "resumes": [
    {"id": "res_001", "label": 1.0, "reason": "High technical relevancy in Agentic workflows (LangGraph). Strong Python/API skills. Offsets minor tenure gap in support."},
    {"id": "res_002", "label": 0.5, "reason": "Superior support experience (7 yrs), but no LLM/GenAI exposure. Partial match for AI Applications role."},
    {"id": "res_003", "label": 1.0, "reason": "Balanced fit: Strong GenAI solutions experience + 4 years customer-facing solutions engineering."},
    {"id": "res_004", "label": 0.5, "reason": "Good tech, but missing GenAI exposure."},
    {"id": "res_005", "label": 0.5, "reason": "Feature dev focus, lacks support depth."},
    {"id": "res_006", "label": 0.5, "reason": "Data Science background, not App Engineer."},
    {"id": "res_007", "label": 0.5, "reason": "Support expert, but thin on coding."},
    {"id": "res_008", "label": 0.0, "reason": "Frontend focus. No stack match."},
    {"id": "res_009", "label": 0.0, "reason": "Java background. Domain mismatch."},
    {"id": "res_010", "label": 0.0, "reason": "Non-technical background."}
  ]
}
```

---

## 6. Data Quality Checklist

- [ ] All resumes are realistic in length and format
- [ ] No personally identifiable information (all fictional)
- [ ] Skills mentioned are real technologies (not made up)
- [ ] Each resume has clear differentiation from others
- [ ] Labels are consistent with rubric criteria
- [ ] Job description includes both required and nice-to-have skills

---

*Document awaiting review before implementation.*

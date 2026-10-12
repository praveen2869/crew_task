import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def create_security_pdf(output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=54,
        leftMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=12
    )
    h2_style = ParagraphStyle(
        'Heading2Custom',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0f766e"),
        spaceBefore=14,
        spaceAfter=8
    )
    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=8
    )
    callout_style = ParagraphStyle(
        'Callout',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e3a8a"),
        spaceBefore=6,
        spaceAfter=10
    )

    story = []

    # PAGE 1: Zero Trust & Perimeter Architecture
    story.append(Paragraph("Enterprise Security Guide & Operating Manual", title_style))
    story.append(Paragraph("Document Classification: RESTRICTED // SEC-OPS-2025-V1 // Page 1", body_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("Chapter 1: Zero Trust Architecture Overview", h2_style))
    story.append(Paragraph(
        "Acme Corp enforces a strict Zero Trust Architecture (ZTA) across all infrastructure zones. "
        "Under the Zero Trust paradigm, no actor, service, network segment, or device is implicitly trusted, "
        "whether originating from inside or outside the physical perimeter. Every access request must be explicitly "
        "authenticated, authorized, and cryptographically verified prior to granting resource access.",
        body_style
    ))
    story.append(Paragraph(
        "Identity verification requires hardware-bound FIDO2 WebAuthn multi-factor authentication (MFA) "
        "for all employees. SMS-based and email-based two-factor authentication methods are strictly prohibited "
        "due to SIM-swapping and adversary-in-the-middle phishing vulnerabilities.",
        body_style
    ))
    story.append(Paragraph(
        "Network micro-segmentation isolates production workloads from staging, QA, and corporate development environments. "
        "Cross-boundary communication is permitted exclusively over mutual TLS (mTLS) with short-lived X.509 certificates.",
        body_style
    ))
    story.append(PageBreak())

    # PAGE 2: Key Management & Credential Lifecycle
    story.append(Paragraph("Chapter 2: Cryptographic Key & API Token Lifecycle", h2_style))
    story.append(Paragraph("Document Classification: RESTRICTED // SEC-OPS-2025-V1 // Page 2", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "All programmatic credentials, service accounts, and API keys must adhere to centralized lifecycle rotation guidelines:",
        body_style
    ))
    story.append(Paragraph(
        "<b>1. Production Service API Keys:</b> Must be cryptographically rotated every 30 days. "
        "Any API key exceeding 45 days of age is automatically deactivated by the automated IAM reaper daemon.",
        callout_style
    ))
    story.append(Paragraph(
        "<b>2. Development and Staging Tokens:</b> Must be rotated every 60 days. "
        "Hardcoding credentials or API keys in source code repositories is categorized as a Tier-1 security violation.",
        callout_style
    ))
    story.append(Paragraph(
        "<b>3. Key Custodianship:</b> Master encryption keys (KEKs) must be housed in dedicated FIPS 140-2 Level 3 "
        "Hardware Security Modules (HSMs). Envelope encryption using AWS KMS or HashiCorp Vault is mandatory for all data at rest.",
        body_style
    ))
    story.append(Paragraph(
        "Emergency Key Revocation: In the event of a suspected leak or credential exposure, security operators must execute "
        "the command 'sec-ops token revoke --all-active' within 5 minutes of incident declaration.",
        body_style
    ))
    story.append(PageBreak())

    # PAGE 3: Incident Response & SLAs
    story.append(Paragraph("Chapter 3: Incident Response Classifications & SLAs", h2_style))
    story.append(Paragraph("Document Classification: RESTRICTED // SEC-OPS-2025-V1 // Page 3", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Security incidents are triaged into three severity tiers based on blast radius and threat impact:",
        body_style
    ))
    story.append(Paragraph(
        "<b>Severity 1 (Critical / Data Compromise):</b> Unauthorized access to customer PII, active ransomware, "
        "or compromise of production root keys. Maximum initial response SLA is exactly 15 minutes. "
        "Executive leadership and legal counsel must be notified within 60 minutes.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Severity 2 (High / Service Disruption):</b> Targeted DDoS attack, single-tenant data isolation failure, "
        "or localized service disruption without data exfiltration. Maximum initial response SLA is 60 minutes.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Severity 3 (Medium / Vulnerability Finding):</b> High-risk CVE discovered in non-critical library or "
        "misconfigured cloud storage bucket. Response SLA is 24 hours, with remediation SLA of 7 calendar days.",
        body_style
    ))
    story.append(PageBreak())

    # PAGE 4: Compliance & Encryption Standards
    story.append(Paragraph("Chapter 4: Regulatory Compliance & Cryptographic Standards", h2_style))
    story.append(Paragraph("Document Classification: RESTRICTED // SEC-OPS-2025-V1 // Page 4", body_style))
    story.append(Spacer(1, 10))
    story.append(Paragraph(
        "Acme Corp maintains annual certification under SOC 2 Type II, ISO 27001:2022, and HIPAA Security Rules. "
        "Independent SOC 2 surveillance audits occur bi-annually every June and December.",
        body_style
    ))
    story.append(Paragraph(
        "Approved Cryptographic Algorithms for Data at Rest and in Transit:",
        body_style
    ))
    story.append(Paragraph(
        "• Symmetric Data Encryption: AES-256 in Galois/Counter Mode (AES-256-GCM) with 96-bit initialization vectors.<br/>"
        "• Asymmetric Signatures: ECDSA using NIST P-384 curves or RSA-4096.<br/>"
        "• Hashing Algorithms: SHA-384 or SHA-512. MD5 and SHA-1 are strictly forbidden in all systems.<br/>"
        "• In-Transit Protocol: Transport Layer Security (TLS) version 1.3 exclusively. TLS 1.0, 1.1, and 1.2 are disabled on all endpoints.",
        body_style
    ))
    story.append(Paragraph(
        "Audit Logging: All privileged administrative actions generate cryptographically signed audit logs stored in an immutable WORM "
        "(Write Once, Read Many) S3 bucket with a compulsory retention period of 7 years.",
        body_style
    ))

    doc.build(story)
    print(f"Successfully generated {output_path}")


if __name__ == "__main__":
    create_security_pdf("sample_docs/enterprise_security_guide.pdf")

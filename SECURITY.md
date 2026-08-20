# Security Policy

SENTINEL is alpha research software. It has not been independently audited or
certified, and this policy does not create a bug bounty, service-level
agreement, or guarantee of payment.

## Supported version

Security fixes target the current `main` branch. Please reproduce an issue
against the latest commit before reporting it.

## Report privately

Use GitHub's private vulnerability reporting form:

https://github.com/noah-ing/SENTINEL/security/advisories/new

If the form is unavailable, open a public issue asking for a private reporting
channel, but do not include exploit details, credentials, sensitive data, or a
working bypass in that issue.

Useful reports include a minimal reproduction, affected commit and component,
expected and observed policy outcome, impact, preconditions, and any suggested
remediation. High-value areas include parser or canonicalization discrepancies,
policy bypasses, unsafe framework-adapter behavior, replay or state-isolation
failures, and accidental disclosure of protected content or credentials.

The benchmark's missed cases, documented mock-model limitations, and ordinary
classifier false positives are evaluation results rather than vulnerabilities
unless they expose a separate security boundary failure.

Do not test against systems, accounts, agents, or data you do not own or have
written permission to assess. Do not exfiltrate data, degrade a service, or
retain sensitive material beyond what is necessary to demonstrate the issue.

Please allow a reasonable remediation window before public disclosure.

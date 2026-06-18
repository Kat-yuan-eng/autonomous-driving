# Security Policy

## Supported Versions

Use this section to tell people about which versions of your project are
currently being supported with security updates.

| Version | Supported          |
|---------|--------------------|
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of the autonomous driving project seriously. If you
discover a security vulnerability, please follow these steps:

1. **Do NOT open a public GitHub issue.**
2. Email the maintainer at `kat-yuan-eng@github` with a description of the
   vulnerability, the steps to reproduce, and the potential impact.
3. You should receive an acknowledgement within 48 hours.
4. Once the vulnerability is verified, we will work on a fix and coordinate
   a disclosure timeline with you.

## Disclosure Policy

- Vulnerabilities will be disclosed via GitHub Security Advisories after
  a fix is available.
- We credit reporters who responsibly disclose vulnerabilities (unless they
  prefer to remain anonymous).

## Scope

The following are considered security vulnerabilities:
- Remote code execution via crafted input data (e.g., malicious map files)
- Denial of service via malformed sensor input
- Authentication bypass in any network-facing component

The following are NOT considered security vulnerabilities:
- Bugs that require physical access to the vehicle
- Issues in third-party dependencies (report upstream)
- Theoretical attacks without a working proof of concept

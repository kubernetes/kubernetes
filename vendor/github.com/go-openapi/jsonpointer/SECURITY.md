# Security Policy

This policy outlines the commitment and practices of the go-openapi maintainers regarding security.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| O.x     | :white_check_mark: |

## Vulnerability checks in place

This repository uses automated vulnerability scans, at every merged commit and at least once a week.

We use:

* [`GitHub CodeQL`][codeql-url]
* [`trivy`][trivy-url]
* [`govulncheck`][govulncheck-url]

Reports are centralized in github security reports and visible only to the maintainers.

## Reporting a vulnerability

If you become aware of a security vulnerability that affects the current repository,
**please report it privately to the maintainers**
rather than opening a publicly visible GitHub issue.

Please follow the instructions provided by github to [Privately report a security vulnerability][github-guidance-url].

> [!NOTE]
> On Github, navigate to the project's "Security" tab then click on "Report a vulnerability".

[codeql-url]: https://github.com/github/codeql
[trivy-url]: https://trivy.dev/docs/latest/getting-started
[govulncheck-url]: https://go.dev/blog/govulncheck
[github-guidance-url]: https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability#privately-reporting-a-security-vulnerability

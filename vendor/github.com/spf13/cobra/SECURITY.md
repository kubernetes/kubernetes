# Security Policy

## Reporting a Vulnerability

The `cobra` maintainers take security issues seriously and
we appreciate your efforts to _**responsibly**_ disclose your findings.
We will make every effort to swiftly respond and address concerns.

To report a security vulnerability:

1. **DO NOT** create a public GitHub issue for the vulnerability!
2. **DO NOT** create a public GitHub Pull Request with a fix for the vulnerability!
3. Send an email to `cobra-security@googlegroups.com`.
4. Include the following details in your report:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact of the vulnerability (to your downstream project, to the Go ecosystem, etc.)
   - Any potential mitigations you've already identified
5. Allow up to 7 days for an initial response.
   You should receive an acknowledgment of your report and an estimated timeline for a fix.
6. (Optional) If you have a fix and would like to contribute your patch, please work
   directly with the maintainers via `cobra-security@googlegroups.com` to
   coordinate pushing the patch to GitHub, cutting a new release, and disclosing the change.

## Response Process

When a security vulnerability report is received, the `cobra` maintainers will:

1. Confirm receipt of the vulnerability report within 7 days.
2. Assess the report to determine if it constitutes a security vulnerability.
3. If confirmed, assign the vulnerability a severity level and create a timeline for addressing it.
4. Develop and test a fix.
5. Patch the vulnerability and make a new GitHub release: the maintainers will coordinate disclosure with the reporter.
6. Create a new GitHub Security Advisory to inform the broader Go ecosystem

## Disclosure Policy

The `cobra` maintainers follow a coordinated disclosure process:

1. Security vulnerabilities will be addressed as quickly as possible.
2. A CVE (Common Vulnerabilities and Exposures) identifier will be requested for significant vulnerabilities
   that are within `cobra` itself.
3. Once a fix is ready, the maintainers will:
   - Release a new version containing the fix.
   - Update the security advisory with details about the vulnerability.
   - Credit the reporter (unless they wish to remain anonymous).
   - Credit the fixer (unless they wish to remain anonymous, this may be the same as the reporter).
   - Announce the vulnerability through appropriate channels
     (GitHub Security Advisory, mailing lists, GitHub Releases, etc.)

## Supported Versions

Security fixes will typically only be released for the most recent major release.

## Upstream Security Issues

`cobra` generally will not accept vulnerability reports that originate in upstream
dependencies. I.e., if there is a problem in Go code that `cobra` depends on,
it is best to engage that project's maintainers and owners.

This security policy primarily pertains only to `cobra` itself but if you believe you've
identified a problem that originates in an upstream dependency and is being widely
distributed by `cobra`, please follow the disclosure procedure above: the `cobra`
maintainers will work with you to determine the severity and ecosystem impact.

## Security Updates and CVEs

Information about known security vulnerabilities and CVEs affecting `cobra` will
be published as GitHub Security Advisories at
https://github.com/spf13/cobra/security/advisories.

All users are encouraged to watch the repository and upgrade promptly when
security releases are published.

## `cobra` Security Best Practices for Users

When using `cobra` in your CLIs, the `cobra` maintainers recommend the following:

1. Always use the latest version of `cobra`.
2. [Use Go modules](https://go.dev/blog/using-go-modules) for dependency management.
3. Always use the latest possible version of Go.

## Security Best Practices for Contributors

When contributing to `cobra`:

1. Be mindful of security implications when adding new features or modifying existing ones.
2. Be aware of `cobra`'s extremely large reach: it is used in nearly every Go CLI
   (like Kubernetes, Docker, Prometheus, etc. etc.)
3. Write tests that explicitly cover edge cases and potential issues.
4. If you discover a security issue while working on `cobra`, please report it
   following the process above rather than opening a public pull request or issue that
   addresses the vulnerability.
5. Take personal sec-ops seriously and secure your GitHub account: use [two-factor authentication](https://docs.github.com/en/authentication/securing-your-account-with-two-factor-authentication-2fa),
   [sign your commits with a GPG or SSH key](https://docs.github.com/en/authentication/managing-commit-signature-verification/about-commit-signature-verification),
   etc.

## Acknowledgments

The `cobra` maintainers would like to thank all security researchers and
community members who help keep cobra, its users, and the entire Go ecosystem secure through responsible disclosures!!

---

*This security policy is inspired by the [Open Web Application Security Project (OWASP)](https://owasp.org/) guidelines and security best practices.*

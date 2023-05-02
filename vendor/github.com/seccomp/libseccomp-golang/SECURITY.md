The libseccomp-golang Security Vulnerability Handling Process
===============================================================================
https://github.com/seccomp/libseccomp-golang

This document document attempts to describe the processes through which
sensitive security relevant bugs can be responsibly disclosed to the
libseccomp-golang project and how the project maintainers should handle these
reports.  Just like the other libseccomp-golang process documents, this
document should be treated as a guiding document and not a hard, unyielding set
of regulations; the bug reporters and project maintainers are encouraged to
work together to address the issues as best they can, in a manner which works
best for all parties involved.

### Reporting Problems

Problems with the libseccomp-golang library that are not suitable for immediate
public disclosure should be emailed to the current libseccomp-golang
maintainers, the list is below.  We typically request at most a 90 day time
period to address the issue before it is made public, but we will make every
effort to address the issue as quickly as possible and shorten the disclosure
window.

* Paul Moore, paul@paul-moore.com
* Tom Hromatka, tom.hromatka@oracle.com
* Kir Kolyshkin, kolyshkin@gmail.com

### Resolving Sensitive Security Issues

Upon disclosure of a bug, the maintainers should work together to investigate
the problem and decide on a solution.  In order to prevent an early disclosure
of the problem, those working on the solution should do so privately and
outside of the traditional libseccomp-golang development practices.  One
possible solution to this is to leverage the GitHub "Security" functionality to
create a private development fork that can be shared among the maintainers, and
optionally the reporter.  A placeholder GitHub issue may be created, but
details should remain extremely limited until such time as the problem has been
fixed and responsibly disclosed.  If a CVE, or other tag, has been assigned to
the problem, the GitHub issue title should include the vulnerability tag once
the problem has been disclosed.

### Public Disclosure

Whenever possible, responsible reporting and patching practices should be
followed, including notification to the linux-distros and oss-security mailing
lists.

* https://oss-security.openwall.org/wiki/mailing-lists/distros
* https://oss-security.openwall.org/wiki/mailing-lists/oss-security

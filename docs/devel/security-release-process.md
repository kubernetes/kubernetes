# Security Release Process

Kubernetes is a large growing community of volunteers, users, and vendors. The Kubernetes community has adopted this security disclosures and response policy to ensure we responsibly handle critical issues.

## Product Security Team (PST)

By their nature security vulnerabilities should be handled quickly and sometimes privately. The Product Security Team is responsible for running the communication, disclosure, and CVE/patch processes. They are not solely responsible for fixing issues and can loop in necessary engineers to test and verify the fixes.

The initial Product Security Team will consist of five volunteers who are all subscribed to the private [Kubernetes Security](https://groups.google.com/forum/#!forum/kubernetes-security) list. These are the people who have been involved in the initial 

- Brandon Philips <brandon.philips> [4096R/154343260542DF34]
- Jess Frazelle <jessfraz@google.com>
- Eric Tune <etune@google.com>
- Jordan Liggitt <jliggitt@redhat.com>
- Matthew Garrett <mjg59@coreos.com> [4096R/BE99F8F0AE90F416] (temporarily advising because he has been on boards like this in the past)

**Known issues**

- We haven't specified a way to cycle the Product Security Team; but we need this process deployed quickly as our current process isn't working. I (@philips) will put a deadline of March 1st 2017 to sort that.

## Disclosures

### Private Disclosure Processes

We ask that suspected vulnerabilities should be responsibility disclosed via our disclosure process. Please visit [http://kubernetes.io/security](http://kubernetes.io/security] to learn more.

### Public Disclosure Processes

If a security issue has been publicly disclosed please IMMEDIATELY email [kubernetes-security@googlegroups.com](kubernetes-security@googlegroups.com) to inform the Product Security Team about the vulnerability and start the patch, release, and communication process.

Next, ask the reporter if the issue can be handled via the private disclosure process. If the reporter denies it is best to proceed publicly and swiftly with the fix and release process. In extreme cases you can ask GitHub to delete the issue but this generally shouldn’t be necessary and is unlikely to make a public disclosure less damaging.

## Patch, Release, and Public Communication

- Request CVE from [DWF](https://github.com/distributedweaknessfiling/DWF-Documentation/) (for embargoed issues) or [oss-security](http://www.openwall.com/lists/oss-security/) (for public issues)
- Set the issue priority based on rough criteria
  - P0 - **Critical impact**, flaws that could be easily exploited by a remote unauthenticated attacker and lead to system compromise (arbitrary code execution) without requiring user interaction. Flaws that require an authenticated remote user, a local user, or an unlikely configuration are not classed as Critical impact.
  - P1 - **Important impact**, flaws that can easily compromise the confidentiality, integrity, or availability of resources. These are the types of vulnerabilities that allow local users to gain privileges, allow unauthenticated remote users to view resources that should otherwise be protected by authentication, allow authenticated remote users to execute arbitrary code, or allow remote users to cause a denial of service.
  - P2 - **Moderate impact**, flaws that may be more difficult to exploit but could still lead to some compromise of the confidentiality, integrity, or availability of resources, under certain circumstances. These are the types of vulnerabilities that could have had a Critical impact or Important impact but are less easily exploited based on a technical evaluation of the flaw, or affect unlikely configurations.
  - P3 - **Low Impact**, other issues that have a security impact. These are the types of vulnerabilities that are believed to require unlikely circumstances to be able to be exploited, or where a successful exploit would give minimal consequences.
- Invite relevant developers to have access to the private security fix repo.
- Work on a fix in a private repo, you can keep track of the issues on this repo as well. Put CVE number in commit and changelog. Please note although CVE is nice, code patch is nicer, better to move ahead without a CVE then to delay in waiting for a CVE.
- Make sure the CVE is mentioned in the commit log and changelog. Changelog for sure, commit log is a “nice to have”, also note situations where security have been fixed and not recognized as such until after the fact
- Apply the fixes to the release branch and any other release branches you will backport to
  - Get LGTM on patches on the private repo
  - **User disclosure** (1-5 days)
    - Email kubernetes-announce@googlegroups.com informing users that a security vulnerability has been disclosed and that a fix will be made available at YYYY-MM-DD HH:MM in the future via this list.
    - Communicate any mitigating steps they can take until a fix is available
  - **Private distributors announce** (1-5 days): 
    - If the issue is Important or Critical email a patch to kubernetes-distributors-announce@googlegroups.com so distributors can prepare builds to be available to users on the day of the issue's announcement. Distributors can ask to be added to this list by emailing kubernetes-security@googlegroups.com and it is up to the Product Security Team's discretion to manage the list.
    - **What if a vendor breaks embargo?** Sometimes could be small or big, just assess the damage and if you need to release earlier because of it, do that, otherwise continue with the plan. Generally speaking when that happens you just push forward and go public ASAP
  - **On release day**
    - Rebase the branch[es] with the fixes, including any other additional branches you applied or backported patches on, with the specific branch on the public repo you will eventually have this in, nothing should have changed upstream unless there were any public cherrypicks.
    - Run the release on these branches (release branch, any additional backported release branches).
    - Make sure all the binaries are up, publicly available, and functional.
    - Open the patches in a PR on the public repo for each release branch you applied the patches to (1 PR per branch).
    - Merge immediately (you cannot accept changes at this time, even for a typo in the CHANGELOG since it would change the git sha of the already built and published release[s]).
    - Cherry-pick the same patches onto the master branch from the release branch. LGTM and merge.
    - At this point everything is public.
    - Email kubernetes-{dev,users,announce,etc}@googlegroups.com to get wide distribution and user action.
    - Remove developers who developed the fix from the private security repo

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/security-release-process.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

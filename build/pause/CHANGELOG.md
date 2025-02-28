# 3.10.1

* Updating base image for Windows pause container to `mcr.microsoft.com/oss/kubernetes/windows-pause-image-base:v0.4.1` to pick up security fixes in the nanoserver base image. ([#130102](https://github.com/kubernetes/kubernetes/pull/130102), [@marosset](https://github.com/marosset))

# 3.10

* Add support for the -v flag on Windows. It prints the version similarly to Linux. ([#125067](https://github.com/kubernetes/kubernetes/pull/125067), [@neolit123](https://github.com/neolit123))

# 3.9

* Unsupported Windows Semi-Annual Channel container images removed (OS Versions removed: 20H2). ([#112924](https://github.com/kubernetes/kubernetes/pull/112924), [@marosset](https://github.com/marosset))

# 3.8

* Updating base image for Windows container images from nanoserver to `mcr.microsoft.com/oss/kubernetes/windows-pause-image-base` which gets built on a Windows machine.
This addresses [Cannot modify registry keys during Windows pause image build process #109161](https://github.com/kubernetes/kubernetes/issues/109161)

# 3.7

* Unsupported Windows Semi-Annual Channel container images removed (OS Versions removed: 1903, 1909, 2004). ([#107056](https://github.com/kubernetes/kubernetes/pull/107056), [@jsturtevant](https://github.com/jsturtevant/))

# 3.6

* Support for Windows container images (OS Versions: 2022) was added. ([#104438](https://github.com/kubernetes/kubernetes/pull/104438), [@nick5616](https://github.com/nick5616))

# 3.5

* Run the container image as non root user per default ([#97963](https://github.com/kubernetes/kubernetes/pull/97963))

# 3.4.1

* Support for Windows container images (OS Versions: 20H2) was added.([#97322](https://prs.k8s.io/97322), [@claudiubelu](https://github.com/claudiubelu))

# 3.4

* Support for Windows container images (OS Versions: 1809, 1903, 1909, 2004) was added. ([#91452](https://prs.k8s.io/91452), [@claudiubelu](https://github.com/claudiubelu))

# 3.3

* update debian-base version to v2.1.0 ([#90665](https://prs.k8s.io/90665), [@justaugustus]

# 3.2

* The pause container is built with the correct "Architecture" metadata. ([#87954](https://prs.k8s.io/87954), [@BenTheElder](https://github.com/BenTheElder))

# 3.1

* The pause container gains a signal handler to clean up orphaned zombie processes. ([#36853](https://prs.k8s.io/36853), [@verb](https://github.com/verb))
* `pause -v` will return build information for the pause binary. ([#56762](https://prs.k8s.io/56762), [@verb](https://github.com/verb))

# 3.0

* The pause container was rewritten entirely in C. ([#23009](https://prs.k8s.io/23009), [@uluyol](https://github.com/uluyol))

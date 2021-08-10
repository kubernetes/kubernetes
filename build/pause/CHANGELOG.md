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

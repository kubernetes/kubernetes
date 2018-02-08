# 3.1

* The pause container gains a signal handler to clean up orphaned zombie processes. ([#36853](https://prs.k8s.io/36853), [@verb](https://github.com/verb))
* `pause -v` will return build information for the pause binary. ([#56762](https://prs.k8s.io/56762), [@verb](https://github.com/verb))

# 3.0

* The pause container was rewritten entirely in C. ([#23009](https://prs.k8s.io/23009), [@uluyol](https://github.com/uluyol))

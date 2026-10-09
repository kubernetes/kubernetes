# Node e2e configuration

Sig-testing migrated node e2e jobs from Jenkins to [Prow], and the node e2e
image config files to [test-infra].

`cos-init-live-restore.yaml` in this directory is still read by node e2e jobs
in [test-infra], which refer to it by path, so please check those jobs when
you change it.

If you have any questions, please contact the approvers of this directory or
#sig-testing.


## Test-infra Links:
Here's where the existing node e2e job config live:

[Image config files](https://github.com/kubernetes/test-infra/tree/master/jobs/e2e_node)

[Node test job args (.properties equivalent)](https://github.com/kubernetes/test-infra/tree/master/config/jobs/kubernetes/sig-node)


[test-infra]: https://github.com/kubernetes/test-infra
[Prow]: https://github.com/kubernetes/test-infra/tree/master/prow

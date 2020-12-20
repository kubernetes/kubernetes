## This directory is deprecated!

Log dumping utility was ported from kubernetes/kubernetes repository to
[kubernetes/test-infra](https://github.com/kubernetes/test-infra/tree/master/logexporter/cluster).
If you require changes to this script, please consider migrating your jobs to use the new
log dumping mechanism first.

Currently, `log-dump.sh` file is added to every newly released `kubekins-e2e` image.
In order to leverage that script, add `USE_TEST_INFRA_LOG_DUMPING` environment variable
to your test job and set its value to `true`.

## Migration steps

For the time being, only GCE and GKE providers are supported by the log-dump mechanism.
To make the mechanism support your Kubernetes provider in tests using `kubekins-e2e`, modify
the `logDumpPath` function in
[kubetest](https://github.com/kubernetes/test-infra/tree/master/kubetest) to handle your provider and
adapt [log-dump.sh](https://github.com/kubernetes/test-infra/blob/master/logexporter/cluster/log-dump.sh)
in accord to your needs.

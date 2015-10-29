This directory forms the base of the main SaltStack configuration. The
place to start with any SaltStack configuration is
[top.sls](top.sls). However, unless you are particularly keen on
reading Jinja templates, the following tables break down what
configurations run on what providers. (NB: The [_states](_states/)
directory is a special directory included by Salt for `ensure` blocks,
and is only used for the [docker](docker/) config.)

Key: M = Config applies to master, n = config applies to nodes

Config                                              | GCE   | Vagrant | AWS |
----------------------------------------------------|-------|---------|-----|
[cadvisor](cadvisor/)                               | M n   | M n     | M n |
[debian-auto-upgrades](debian-auto-upgrades/)       | M n   | M n     | M n |
[docker](docker/)                                   | M n   | M n     | M n |
[etcd](etcd/)                                       | M     | M       | M   |
[fluentd-es](fluentd-es/) (pillar conditional)      | M n   | M n     | M n |
[fluentd-gcp](fluentd-gcp/) (pillar conditional)    | M n   | M n     | M n |
[generate-cert](generate-cert/)                     | M     | M       | M   |
[kube-addons](kube-addons/)                         | M     | M       | M   |
[kube-apiserver](kube-apiserver/)                   | M     | M       | M   |
[kube-controller-manager](kube-controller-manager/) | M     | M       | M   |
[kube-proxy](kube-proxy/)                           |   n   |   n     |   n |
[kube-scheduler](kube-scheduler/)                   | M     | M       | M   |
[kubelet](kubelet/)                                 | M n   | M n     | M n |
[logrotate](logrotate/)                             | M n   |   n     | M n |
[monit](monit/)                                     | M n   | M n     | M n |
[nginx](nginx/)                                     | M     | M       | M   |
[static-routes](static-routes/) (vsphere only)      |       |         |     |
[base](base.sls)                                    | M n   | M n     | M n |
[kube-client-tools](kube-client-tools.sls)          | M     | M       | M   |


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/saltbase/salt/README.md?pixel)]()

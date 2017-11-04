# SaltStack configuration

This is the root of the SaltStack configuration for Kubernetes. A high
level overview for the Kubernetes SaltStack configuration can be found [in the docs tree.](https://kubernetes.io/docs/admin/salt/)

This SaltStack configuration currently applies to default
configurations for Debian-on-GCE, Fedora-on-Vagrant, Ubuntu-on-AWS and
Ubuntu-on-Azure. (That doesn't mean it can't be made to apply to an
arbitrary configuration, but those are only the in-tree OS/IaaS
combinations supported today.) As you peruse the configuration, these
are shorthanded as `gce`, `vagrant`, `aws`, `azure-legacy` in `grains.cloud`;
the documentation in this tree uses this same shorthand for convenience.

See more:
* [pillar](pillar/)
* [reactor](reactor/)
* [salt](salt/)


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/saltbase/README.md?pixel)]()

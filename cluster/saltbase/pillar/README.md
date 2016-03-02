The
[SaltStack pillar](http://docs.saltstack.com/en/latest/topics/pillar/)
data is partially statically derived from the contents of this
directory. The bulk of the pillars are hard to perceive from browsing
this directory, though, because they are written into
[cluster-params.sls](cluster-params.sls) at cluster inception.

* [cluster-params.sls](cluster-params.sls) is generated entirely at cluster inception. See e.g. [configure-vm.sh](../../gce/configure-vm.sh#L262)
* [docker-images.sls](docker-images.sls) stores the Docker tags of the current Docker-wrapped server binaries, twiddling by the Salt install script
* [logging.sls](logging.sls) defines the cluster log level
* [mine.sls](mine.sls): defines the variables shared across machines in the Salt
  mine. It is starting to be largely deprecated in use, and is totally
  unavailable on GCE, which runs standalone.
* [privilege.sls](privilege.sls) defines whether privileged containers are allowed.
* [top.sls](top.sls) defines which pillars are active across the cluster.

## Future work

Document the current pillars across providers


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/saltbase/pillar/README.md?pixel)]()

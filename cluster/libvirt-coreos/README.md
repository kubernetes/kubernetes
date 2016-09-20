# Kubernetes CoreOS cluster

With this tutorial one creates a Kubernetes CoreOS cluster containing of one
master and three minions (workers) running on `192.168.10.1`-`192.168.10.4`.

For working correctly you need to create the directory addressed as `POOL_PATH` in
`util.sh`:
```
$ sudo mkdir /var/lib/libvirt/images/kubernetes
$ sudo chown -R $USER:$USER /var/lib/libvirt/images/kubernetes/
```

Then we follow the instructions in the main `kubernetes` directory.

For debugging set `export UTIL_SH_DEBUG=1`.
```
$ export KUBERNETES_PROVIDER=libvirt-coreos
$ make release-skip-tests
$ ./cluster/kube-up.sh
```

To bring the cluster down again, execute:
```
$ ./cluster/kube-down.sh
```

Have fun!



[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/libvirt-coreos/README.md?pixel)]()


Fixed some libvirt volume mess.

For proper working you need to create the directory addressed as `POOL_PATH` in
`util.sh`


```
sudo mkdir /var/lib/libvirt/images/kubernetes
``` 

Then follow the instructions in the main `kubernetes` directory.

```
export KUBERNETES_PROVIDER=libvirt-coreos
cluster/kube-up.sh 
```

For debugging set `export UTIL_SH_DEBUG=1`.

Have fun!
# Kubernetes-Server

This image consists of the apiserver and controller-manager.

You need to point it to all your Docker daemons and etcd nodes:

```
docker run kubernetes-server http://etcd1:4001 http://etcd2:4001 [http://...] hostname1 hostname2 [hostname...]
```

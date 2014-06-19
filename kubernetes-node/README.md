# Kubernetes-Node

This image consists of etcd to persist the scheduler state and kubelet to talk
to Docker. This should be deployed to all your Docker nodes. You need to
provide the node's external IP and a cluster discovery endpoint. Get a new
cluster discovery endpoint [here](https://discovery.etcd.io/new).
Since kubelet needs to talk to Docker, you need to also bind-mount
`/var/run/docker.sock` to your container:


```
docker run -v /var/run/docker.sock:/docker.sock -d -P kubernetes-node external-ip:7001 external-ip:4001 etcd-cluster-discovery-endpoint
```

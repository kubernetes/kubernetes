# Cluster Troubleshooting
Most of the time, if you encounter problems, it is your application that is having problems.  For application
problems please see the [application troubleshooting guide](application-troubleshooting.md).

## Listing your cluster
The first thing to debug in your cluster is if your nodes are all registered correctly.

Run
```
kubectl get nodes
```

And verify that all of the nodes you expect to see are present and that they are all in the ```Ready``` state.

## Looking at logs
For now, digging deeper into the cluster requires logging into the relevant machines.  Here are the locations
of the relevant log files.  (note that on systemd based systems, you may need to use ```journalctl``` instead)

### Master
   * /var/log/kube-apiserver.log - API Server, responsible for serving the API
   * /var/log/kube-scheduler.log - Scheduler, responsible for making scheduling decisions
   * /var/log/kube-controller-manager.log - Controller that manages replication controllers

### Worker Nodes
   * /var/log/kubelet.log - Kubelet, responsible for running containers on the node
   * /var/log/kube-proxy.log - Kube Proxy, responsible for service load balancing



[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/cluster-troubleshooting.md?pixel)]()

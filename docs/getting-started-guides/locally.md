## Getting started locally

In a separate tab of your terminal, run:

```
cd kubernetes
hack/local-up-cluster.sh
```

This will build and start a lightweight local cluster, consisting of a master and a single minion. Type Control-C to shut it down.

If you are running both a remote kubernetes cluster and the local cluster, you can determine which you talk to using the ```KUBERNETES_MASTER``` environment variable.

# Accessing the Cluster

## Using the Kubernetes proxy to access the cluster
Information about the cluster can be accessed by using a proxy URL and by providing the keys to the cluster.
For example, for a cluster that has cluster-level logging enabled using Elasticsearch you can fetch information about
the Elasticsearch logging cluster.

First, you will need to obtain the keys (username and password) for your cluster:

```
$ cat ~/.kube/kubernetes-satnam2_kubernetes/kubernetes_auth
{
  "User": "admin",
  "Password": "4mty0Vl9nNFfwLJz",
  "CAFile": "/Users/satnam/.kube/kubernetes-satnam2_kubernetes/kubernetes.ca.crt",
  "CertFile": "/Users/satnam/.kube/kubernetes-satnam2_kubernetes/kubecfg.crt",
  "KeyFile": "/Users/satnam/.kube/kubernetes-satnam2_kubernetes/kubecfg.key"
}
```

To access a service endpoint `/alpha/beta/gamma/` via the proxy service for your service `myservice` you need to specify an HTTPS address
for the Kubernetes master followed by `/api/v1beta1/proxy/services/myservice/alpha/beta/gamma/`. Currently it is important to
specify the trailing `/`.

Here is a list of representative cluster-level system services:
```
$ kubectl get services --selector="kubernetes.io/cluster-service=true"
NAME                    LABELS                                                          SELECTOR                     IP                  PORT
elasticsearch-logging   kubernetes.io/cluster-service=true,name=elasticsearch-logging   name=elasticsearch-logging   10.0.251.46         9200
kibana-logging          kubernetes.io/cluster-service=true,name=kibana-logging          name=kibana-logging          10.0.118.199        5601
kube-dns                k8s-app=kube-dns,kubernetes.io/cluster-service=true             k8s-app=kube-dns             10.0.0.10           53
monitoring-grafana      kubernetes.io/cluster-service=true,name=grafana                 name=influxGrafana           10.0.15.119         80
monitoring-heapster     kubernetes.io/cluster-service=true,name=heapster                name=heapster                10.0.101.222        80
monitoring-influxdb     kubernetes.io/cluster-service=true,name=influxdb                name=influxGrafana           10.0.155.212        80
```

Using this information you can now issue the following `curl` command to get status information about
the Elasticsearch logging service.
```
$ curl -k -u admin:4mty0Vl9nNFfwLJz https://104.197.5.247/api/v1beta1/proxy/services/elasticsearch-logging/
{
  "status" : 200,
  "name" : "Senator Robert Kelly",
  "cluster_name" : "kubernetes_logging",
  "version" : {
    "number" : "1.4.4",
    "build_hash" : "c88f77ffc81301dfa9dfd81ca2232f09588bd512",
    "build_timestamp" : "2015-02-19T13:05:36Z",
    "build_snapshot" : false,
    "lucene_version" : "4.10.3"
  },
  "tagline" : "You Know, for Search"
}
```

You can provide a suffix and parameters:
```
$ curl -k -u admin:4mty0Vl9nNFfwLJz https://104.197.5.247/api/v1beta1/proxy/services/elasticsearch-logging/_cluster/health?pretty=true
{
  "cluster_name" : "kubernetes_logging",
  "status" : "yellow",
  "timed_out" : false,
  "number_of_nodes" : 1,
  "number_of_data_nodes" : 1,
  "active_primary_shards" : 5,
  "active_shards" : 5,
  "relocating_shards" : 0,
  "initializing_shards" : 0,
  "unassigned_shards" : 5
}
```

You can also visit the endpoint of a service via the proxy URL e.g.
```
https://104.197.5.247/api/v1beta1/proxy/services/kibana-logging/
```
The first time you access the cluster using a proxy address from a browser you will be prompted
for a username and password which can also be found in the `User` and `Password` fields of the `kubernetes_auth`
file.

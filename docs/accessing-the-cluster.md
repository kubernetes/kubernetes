# Accessing the Cluster

## Using the LMKTFY proxy to access the cluster
Information about the cluster can be accessed by using a proxy URL and by providing the keys to the cluster.
For example, for a cluster that has cluster-level logging enabled using Elasticsearch you can fetch information about
the Elasticsearch logging cluster.

First, you will need to obtain the keys (username and password) for your cluster:

```
$ cat ~/.lmktfy/lmktfy-satnam2_lmktfy/lmktfy_auth
{
  "User": "admin",
  "Password": "4mty0Vl9nNFfwLJz",
  "CAFile": "/Users/satnam/.lmktfy/lmktfy-satnam2_lmktfy/lmktfy.ca.crt",
  "CertFile": "/Users/satnam/.lmktfy/lmktfy-satnam2_lmktfy/lmktfycfg.crt",
  "KeyFile": "/Users/satnam/.lmktfy/lmktfy-satnam2_lmktfy/lmktfycfg.key"
}
```

To access a service endpoint `/alpha/beta/gamma/` via the proxy service for your service `myservice` you need to specify an HTTPS address
for the LMKTFY master followed by `/api/v1beta1/proxy/services/myservice/alpha/beta/gamma/`. Currently it is important to
specify the trailing `/`.

Here is a list of representative cluster-level system services:
```
$ lmktfyctl get services --selector="lmktfy.io/cluster-service=true"
NAME                    LABELS                                                          SELECTOR                     IP                  PORT
elasticsearch-logging   lmktfy.io/cluster-service=true,name=elasticsearch-logging   name=elasticsearch-logging   10.0.251.46         9200
kibana-logging          lmktfy.io/cluster-service=true,name=kibana-logging          name=kibana-logging          10.0.118.199        5601
lmktfy-dns                lmktfy-app=lmktfy-dns,lmktfy.io/cluster-service=true             lmktfy-app=lmktfy-dns             10.0.0.10           53
monitoring-grafana      lmktfy.io/cluster-service=true,name=grafana                 name=influxGrafana           10.0.15.119         80
monitoring-heapster     lmktfy.io/cluster-service=true,name=heapster                name=heapster                10.0.101.222        80
monitoring-influxdb     lmktfy.io/cluster-service=true,name=influxdb                name=influxGrafana           10.0.155.212        80
```

Using this information you can now issue the following `curl` command to get status information about
the Elasticsearch logging service.
```
$ curl -k -u admin:4mty0Vl9nNFfwLJz https://104.197.5.247/api/v1beta1/proxy/services/elasticsearch-logging/
{
  "status" : 200,
  "name" : "Senator Robert Kelly",
  "cluster_name" : "lmktfy_logging",
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
  "cluster_name" : "lmktfy_logging",
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
for a username and password which can also be found in the `User` and `Password` fields of the `lmktfy_auth`
file.

## Redirect
A `redirect` request on a service will return a HTTP redirect response which identifies a specific node that
can handle the request. Since the hostname that is returned is usually only accessible from inside the cluster
this feature is useful only for code running inside the cluster. Subsequent `redirect` calls to the same
resource may return different results e.g. when the service picks different replica nodes to serve the request.
This feature can be useful to short circuit calls to the proxy server by obtaining the address of a node on the
cluster which can be used for further requests which do not involve the proxy server.

For example, the query below is run on
a GCE virtual machine `oban` that is running in the same project and GCE default network as the LMKTFY
cluster. The `-L` flag tells curl to follow the redirect information returned by the redirect call.

```
satnam@oban:~$ curl -L -k -u admin:4mty0Vl9nNFfwLJz https://104.197.5.247/api/v1beta1/redirect/services/elasticsearch-logging/
{
  "status" : 200,
  "name" : "Skin",
  "cluster_name" : "lmktfy_logging",
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

We can examine the actual redirect header:

```
satnam@oban:~$ curl -v -k -u admin:4mty0Vl9nNFfwLJz https://104.197.5.247/api/v1beta1/redirect/services/elasticsearch-logging/
* About to connect() to 104.197.5.247 port 443 (#0)
*   Trying 104.197.5.247...
* connected
* Connected to 104.197.5.247 (104.197.5.247) port 443 (#0)
* successfully set certificate verify locations:
*   CAfile: none
  CApath: /etc/ssl/certs
* SSLv3, TLS handshake, Client hello (1):
* SSLv3, TLS handshake, Server hello (2):
* SSLv3, TLS handshake, CERT (11):
* SSLv3, TLS handshake, Server key exchange (12):
* SSLv3, TLS handshake, Server finished (14):
* SSLv3, TLS handshake, Client key exchange (16):
* SSLv3, TLS change cipher, Client hello (1):
* SSLv3, TLS handshake, Finished (20):
* SSLv3, TLS change cipher, Client hello (1):
* SSLv3, TLS handshake, Finished (20):
* SSL connection using ECDHE-RSA-AES256-GCM-SHA384
* Server certificate:
* 	 subject: CN=lmktfy-master
* 	 start date: 2015-03-04 19:40:24 GMT
* 	 expire date: 2025-03-01 19:40:24 GMT
* 	 issuer: CN=104.197.5.247@1425498024
* 	 SSL certificate verify result: unable to get local issuer certificate (20), continuing anyway.
* Server auth using Basic with user 'admin'
> GET /api/v1beta1/redirect/services/elasticsearch-logging/ HTTP/1.1
> Authorization: Basic YWRtaW46M210eTBWbDluTkZmd0xKeg==
> User-Agent: curl/7.26.0
> Host: 104.197.5.247
> Accept: */*
>
* additional stuff not fine transfer.c:1037: 0 0
* HTTP 1.1 or later with persistent connection, pipelining supported
< HTTP/1.1 307 Temporary Redirect
< Server: nginx/1.2.1
< Date: Thu, 05 Mar 2015 00:14:45 GMT
< Content-Type: text/plain; charset=utf-8
< Content-Length: 0
< Connection: keep-alive
< Location: http://10.244.2.7:9200
<
* Connection #0 to host 104.197.5.247 left intact
* Closing connection #0
* SSLv3, TLS alert, Client hello (1):

```

This shows that the request to `https://104.197.5.247/api/v1beta1/redirect/services/elasticsearch-logging/` is redirected to `http://10.244.2.7:9200`.
If we examine the pods on the cluster we can see that `http://10.244.2.7` is the address of a pod that is running the Elasticsearch service.


```
$ lmktfyctl get pods
POD                                          IP                  CONTAINER(S)            IMAGE(S)                            HOST                                                                  LABELS                                                                      STATUS              CREATED
elasticsearch-logging-controller-gziey       10.244.2.7          elasticsearch-logging   lmktfy/elasticsearch:1.0        lmktfy-minion-hqhv.c.lmktfy-satnam2.internal/104.154.33.252   lmktfy.io/cluster-service=true,name=elasticsearch-logging               Running             5 hours
kibana-logging-controller-ls6k1              10.244.1.9          kibana-logging          lmktfy/kibana:1.1               lmktfy-minion-h5kt.c.lmktfy-satnam2.internal/146.148.80.37    lmktfy.io/cluster-service=true,name=kibana-logging                      Running             5 hours
lmktfy-dns-oh43e                               10.244.1.10         etcd                    quay.io/coreos/etcd:v2.0.3          lmktfy-minion-h5kt.c.lmktfy-satnam2.internal/146.148.80.37    lmktfy-app=lmktfy-dns,lmktfy.io/cluster-service=true,name=lmktfy-dns           Running             5 hours
                                                                 lmktfy2sky                lmktfy/lmktfy2sky:1.0
                                                                 skydns                  lmktfy/skydns:2014-12-23-001
monitoring-heapster-controller-fplln         10.244.0.4          heapster                lmktfy/heapster:v0.8            lmktfy-minion-2il2.c.lmktfy-satnam2.internal/130.211.155.16   lmktfy.io/cluster-service=true,name=heapster,uses=monitoring-influxdb   Running             5 hours
monitoring-influx-grafana-controller-0133o   10.244.3.4          influxdb                lmktfy/heapster_influxdb:v0.3   lmktfy-minion-kmin.c.lmktfy-satnam2.internal/130.211.173.22   lmktfy.io/cluster-service=true,name=influxGrafana                       Running             5 hours
                                                                 grafana                 lmktfy/heapster_grafana:v0.4
```

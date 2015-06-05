# Accessing the Cluster

 * [Using the Kubernetes proxy](#proxy)
 * [Requesting redirects](#redirect)

## <a name="proxy"></a>Using the Kubernetes proxy to access the cluster
Information about the cluster can be accessed by using a proxy URL and the cluster authentication keys.
For example, if a cluster has cluster-level logging enabled (using Elasticsearch), you can retrieve information about the Elasticsearch logging on that cluster through a proxy URL.

### Retrieving the authentcation keys and proxy URLs
Use `kubectl` commands to retrieve the access information.

To retrieve the authentication keys for your clusters, run the following command:
```
$ kubectl config view

  ...
  users:
  - name: kubernetes_logging
    user:
      client-certificate-data: REDACTED
      client-key-data: REDACTED
      token: cvIH2BYtNS85QG0KSLHgl5Oba4YNQOrx
  - name: kubernetes_logging-basic-auth
    user:
      password: 4mty0Vl9nNFfwLJz
      username: admin
```

To retrieve the address of the Kubernetes master cluster and the proxy URLs for services, run the following command:
```
$ kubectl cluster-info

  Kubernetes master is running at https://104.197.5.247
  elasticsearch-logging is running at https://104.197.5.247/api/v1/proxy/namespaces/default/services/elasticsearch-logging
  kibana-logging is running at https://104.197.5.247/api/v1/proxy/namespaces/default/services/kibana-logging
  kube-dns is running at https://104.197.5.247/api/v1/proxy/namespaces/default/services/kube-dns
  grafana is running at https://104.197.5.247/api/v1/proxy/namespaces/default/services/monitoring-grafana
  heapster is running at https://104.197.5.247/api/v1/proxy/namespaces/default/services/monitoring-heapster
```

**Note**: Currently, adding trailing forward slashes '.../' to proxy URLs is required, for example: `https://104.197.5.247/api/v1/proxy/namespaces/default/services/elasticsearch-logging/`.

#### Manually constructing proxy URLs
As mentioned above, you use the `kubectl cluster-info` command to retrieve the service's proxy URL. To create proxy URLs that include service endpoints, suffixes, and parameters, you simply append to the service's proxy URL:  
`http://`*`kubernetes_master_address`*`/`*`service_path`*`/`*`service_name`*`/`*`service_endpoint-suffix-parameter`*
##### Examples
 * To access the Elasticsearch service endpoint `_search?q=user:kimchy`, you would use:   `http://104.197.5.247/api/v1/proxy/namespaces/default/services/elasticsearch-logging/_search?q=user:kimchy`

 * To access the Elasticsearch cluster health information `_cluster/health?pretty=true`, you would use:   `https://104.197.5.247/api/v1/proxy/namespaces/default/services/elasticsearch-logging/_cluster/health?pretty=true`
  ```
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

### Accessing cluster information
You can run `curl` commands or use a web browser to access the information about cluster services. Depending on how secure the information is, you can choose to use basic authentication or token authentication (bearer and insecure). 

#### Using `curl` commands
Run `curl` commands using the following formats:

 * Basic authentication: `$ curl -k -u` *`username`*`:`*`password`* *`proxy_URL`*`/`
 * Token authentication: 
     * Bearer tokens: `curl --insecure -H "Authorization: Bearer` *`access_token`*`"` *`proxy_URL`*`/`
	 * Insecure: `curl --insecure -H "Authorization: Bearer` *`access_token`*`"` *`proxy_URL`*`/`

**Note**: Currently, adding a trailing forward slash '.../' to each proxy URL is required.

For example, to get status information about the Elasticsearch logging service, you would run one of the following commands:

 * Basic authentication:
`$ curl -k -u admin:4mty0Vl9nNFfwLJz https://104.197.5.247/api/v1/proxy/namespaces/default/services/elasticsearch-logging/`

 * Token authentication:
`$ curl -k -H "Authorization: Bearer cvIH2BYtNS85QG0KSLHgl5Oba4YNQOrx" https://104.197.5.247/api/v1/proxy/namespaces/default/services/elasticsearch-logging/`

The result for either authentication method:
```
{
  "status" : 200,
  "name" : "Alaris",
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

#### Using web browsers
In a web browser, navigate to the proxy URL and then enter your username and password when prompted. For example, you would copy and paste the following proxy URL into the address bar of your browser:
```
https://104.197.5.247/api/v1/proxy/namespaces/default/services/elasticsearch-logging/
```

## <a name="redirect"></a>Requesting redirects
Use a `redirect` request so that the server returns an HTTP redirect response and identifies the specific node and service that
can handle the request. 

**Note**: Since the hostname or address that is returned is usually only accessible from inside the cluster,
sending `redirect` requests is useful only for code running inside the cluster. Also, keep in mind that any subsequent `redirect` requests to the same
server might return different results (because another node at that point in time can better serve the request).

**Tip**: Use a redirect request to reduce calls to the proxy server by first obtaining the address of a node on the
cluster and then using that returned address for all subsequent requests.

##### Example
To request a redirect and then verify the address that gets returned, let's run a query on `oban` (Google Compute Engine virtual machine). Note that `oban` is running in the same project and default network (Google Compute Engine) as the Kubernetes cluster. 

To request a redirect for the Elasticsearch service, we can run the following `curl` command:
```
user@oban:~$ curl -L -k -u admin:4mty0Vl9nNFfwLJz https://104.197.5.247/api/v1/redirect/namespaces/default/services/elasticsearch-logging/
{
  "status" : 200,
  "name" : "Skin",
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
**Note**: We use the `-L` flag in the request so that `curl` follows the returned redirect address and retrieves the Elasticsearch service information.

If we examine the actual redirect header (instead run the same `curl` command with `-v`), we see that the request to `https://104.197.5.247/api/v1/redirect/namespaces/default/services/elasticsearch-logging/` is redirected to `http://10.244.2.7:9200`:
```
user@oban:~$ curl -v -k -u admin:4mty0Vl9nNFfwLJz https://104.197.5.247/api/v1/redirect/namespaces/default/services/elasticsearch-logging/
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
* 	 subject: CN=kubernetes-master
* 	 start date: 2015-03-04 19:40:24 GMT
* 	 expire date: 2025-03-01 19:40:24 GMT
* 	 issuer: CN=104.197.5.247@1425498024
* 	 SSL certificate verify result: unable to get local issuer certificate (20), continuing anyway.
* Server auth using Basic with user 'admin'
> GET /api/v1/redirect/namespaces/default/services/elasticsearch-logging HTTP/1.1
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

We can also run the `kubectl get pods` command to view a list of the pods on the cluster and verify that `http://10.244.2.7` is where the Elasticsearch service is running:
```
$ kubectl get pods
POD                                          IP                  CONTAINER(S)            IMAGE(S)                            HOST                                                                  LABELS                                                                      STATUS              CREATED
elasticsearch-logging-controller-gziey       10.244.2.7          elasticsearch-logging   kubernetes/elasticsearch:1.0        kubernetes-minion-hqhv.c.kubernetes-user2.internal/104.154.33.252   kubernetes.io/cluster-service=true,name=elasticsearch-logging               Running             5 hours
kibana-logging-controller-ls6k1              10.244.1.9          kibana-logging          kubernetes/kibana:1.1               kubernetes-minion-h5kt.c.kubernetes-user2.internal/146.148.80.37    kubernetes.io/cluster-service=true,name=kibana-logging                      Running             5 hours
kube-dns-oh43e                               10.244.1.10         etcd                    quay.io/coreos/etcd:v2.0.3          kubernetes-minion-h5kt.c.kubernetes-user2.internal/146.148.80.37    k8s-app=kube-dns,kubernetes.io/cluster-service=true,name=kube-dns           Running             5 hours
                                                                 kube2sky                kubernetes/kube2sky:1.0
                                                                 skydns                  kubernetes/skydns:2014-12-23-001
monitoring-heapster-controller-fplln         10.244.0.4          heapster                kubernetes/heapster:v0.8            kubernetes-minion-2il2.c.kubernetes-user2.internal/130.211.155.16   kubernetes.io/cluster-service=true,name=heapster,uses=monitoring-influxdb   Running             5 hours
monitoring-influx-grafana-controller-0133o   10.244.3.4          influxdb                kubernetes/heapster_influxdb:v0.3   kubernetes-minion-kmin.c.kubernetes-user2.internal/130.211.173.22   kubernetes.io/cluster-service=true,name=influxGrafana                       Running             5 hours
                                                                 grafana                 kubernetes/heapster_grafana:v0.4
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/accessing-the-cluster.md?pixel)]()

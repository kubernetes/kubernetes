# Log Aggregration

The _install_log_agggregation.yml_ playbook configures the kubernetes cluster
to aggregate all the logs from the application containers into an elasticsearch
data store and display them in a kibana front end.

The ELK stack is highly configurable, and there's no one best solution for all systems.
This playbook makes a number of choices which can be adapted in different ways for
systems larger or smaller.

To install, you should have an ansible hosts file that was created when the cluster
was made.
```
cd ansible
hosts_dir=~/.clc_kube/${CLC_CLUSTER_NAME}/hosts/
ansible-playbook -i ${hosts_dir} install_log_aggregation.yml
```

### Elasticsearch

The playbook installs six elasticsearch nodes of three types: one master,
two clients, three data nodes.  It uses a replication controller to manage
each type of node.

#### Access

The client nodes implement the _elasticsearch_logging_ service which is exposed
via a NodePort at 30092. So _curl http://[ANY_NODE_IP]:30092_
will output the standard elasticsearch greeting, something like:

```
{
    "name": "Misfit",
    "cluster_name": "es_in_k8s",
    "version":
    {
        "number": "2.1.1",
        "build_hash": "40e2c53a6b6c2972b3d13846e450e66f4375bd71",
        "build_timestamp": "2015-12-15T13:05:55Z",
        "build_snapshot": false,
        "lucene_version": "5.3.1"
    },
    "tagline": "You Know, for Search"
}
```

See the [elasticsearch api](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html)
for more ways to interact with this endpoint.

#### Storage

Data nodes store data in an _emptyDir_ volume. This is probably reasonable for
small amounts of data which are replicated across other nodes. The actual storage
location of the data is in the docker storage directories.  In our cluster setup,
we relocate it to _/data/var/lib/docker_  and by default allocate 100GB to the
/data mount.  The _emptytDir_ will survive pod restarts, but not pod deletion (or
relocation)

For more durable storage, a persistent volume is highly recommended.  That approach
would require explicitly declaring a storage location per pod and using a
replication controller for each pod.

#### Namespace

Elasticsearch currently lives within the _kube-system_ namespace.  Other uses
of elasticsearch besides logs are possible, and having the cluster inside the
_kube-system_ namespace may not be appropriate in that case.

#### Indexing

For reasonable log searching, we need a template for the _logstash-_ index which
does not analyze fields by default.  Once the elasticsearch cluster is up, there's
a kubernetes job (see _roles/kubernetes-manifest/templates/job-es-template.yml.j2_)
which posts the correct template to the service.  This configuration is done
automatically as part of the playbook.

#### Management

Longer-term management of the cluster involves archiving or deleting older data.
The specifics will depend entirely on the amounts and types of data being recorded,
The Elasticsearch API makes these task reasonable easy to do, and it should prove
simple to implement scheduling using Kubernetes jobs.

### Fluentd

The _fluentd_ pod is run as a DaemonSet so that it runs once and once only on
each minion node. Container log directories on the node are automatically
mounted into the _fluentd_ container, and the logs parsed and sent to the
elasticsearch instance at
http://elasticsearch-logging:9200

### Kibana

_kibana_ also communicates with http://elasticsearch-logging:9200.  It is exposed as a
NodePort at port 30056, so the UI can be accessed at http://[ANY_NODE_IP]:30056.
Searching, manipulating and visualizing the output is highly dependent on the
contents, and in a bare cluster, most or all of the logging will be from the

Please note, it is _not_ possible to access the kibana UI from the proxy-api.
Although (a) _cluster-info_ will report something like
*kibana-logging is running at https://10.141.117.29:6443/api/v1/proxy/namespaces/kube-system/services/kibana-logging*
and (b) running `kubectl proxy -p 8001` should expose that on localhost without
need for client certificates, it doesn't work.  Kibana is a nodejs
application and uses redirects to urls like `/apps/kibana` which are not handled
nicely by kubernetes proxy-api.



## Excessive details

### Finding an address to access a NodePort.  

What if you don't have the IP addresses handy? _kubectl_ could display them.
The _-o template_ option allows one to extract particular pieces of data from the
larger set. It may not be pretty. But if you have been missing the joys of
working with xquery, you might like it.

```
kubectl get node -o template --template='{{ range .items }}{{ range .status.addresses  }}{{ if  eq .type "LegacyHostIP"  }}{{ .address|printf "http://%s:30056\n"  }}{{ end }}{{ end }}{{ end }}'
```

### Inspect what kubernetes is running for log aggregration

Service Accounts:
```
$ kubectl --namespace=kube-system get serviceaccounts
NAME             SECRETS   AGE
elasticsearch    1         45m
```
Replication Controllers:
```
$ kubectl --namespace=kube-system get rc
CONTROLLER                       CONTAINER(S)     IMAGE(S)                                                SELECTOR                              REPLICAS   AGE
es-client                        es-client        quay.io/pires/docker-elasticsearch-kubernetes:2.1.1     component=elasticsearch,role=client   2          45m
es-data                          es-data          quay.io/pires/docker-elasticsearch-kubernetes:2.1.1     component=elasticsearch,role=data     3          45m
es-master                        es-master        quay.io/pires/docker-elasticsearch-kubernetes:2.1.1     component=elasticsearch,role=master   1          45m
kibana-logging                   kibana-logging   kibana:4.3.1                                            component=kibana-logging              1          45m
```
Daemon Sets:
```
$ kubectl --namespace=kube-system get ds
NAME              CONTAINER(S)            IMAGE(S)                                              SELECTOR              NODE-SELECTOR
fluentd-logging   fluentd-elasticsearch   gcr.io/google_containers/fluentd-elasticsearch:1.13   app=fluentd-logging   <none>
```
Services (these IPs are in the virtual service IP network):
```
$ kubectl --namespace=kube-system get svc
NAME                      CLUSTER_IP     EXTERNAL_IP   PORT(S)             SELECTOR                              AGE
elasticsearch             10.0.7.200     nodes         9200/TCP            component=elasticsearch,role=client   46m
elasticsearch-discovery   10.0.126.103   <none>        9300/TCP            component=elasticsearch,role=master   46m
elasticsearch-logging     10.0.179.3     <none>        9200/TCP            component=elasticsearch,role=client   46m
kibana-logging            10.0.139.24    nodes         5601/TCP            component=kibana-logging              45m
```
Jobs:
```
$ kubectl --namespace=kube-system get job
JOB           CONTAINER(S)   IMAGE(S)     SELECTOR        SUCCESSFUL
es-template   curl           tutum/curl   app in (curl)   1
```
Endpoints (these IPs are in the pod network):
```
$ kubectl --namespace=kube-system get ep
NAME                      ENDPOINTS                           AGE
elasticsearch             10.244.71.4:9200,10.244.78.5:9200   48m
elasticsearch-discovery   10.244.78.3:9300                    48m
elasticsearch-logging     10.244.71.4:9200,10.244.78.5:9200   47m
kibana-logging            10.244.78.6:5601                    47m
```

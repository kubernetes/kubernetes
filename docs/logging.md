# Logging

## Logging by Kubernetes Components
Kubernetes components, such as kubelet and apiserver, use the [glog](https://godoc.org/github.com/golang/glog) logging library.  Developer conventions for logging severity are described in [devel/logging.md](devel/logging.md).

## Logging in Containers
There are no Kubernetes-specific requirements for logging from within containers. [search](https://www.google.com/?q=docker+container+logging) will turn up any number of articles about logging and
Docker containers.  However, we do provide an example of how to collect, index, and view pod logs [using Fluentd, Elasticsearch, and Kibana](./getting-started-guides/logging.md)


## Logging to Elasticsearch on the GCE platform
Currently the collection of container logs using the [Fluentd](http://www.fluentd.org/) log collector is 
enabled by default for clusters created for the GCE platform. Each node uses Fluentd to collect
the container logs which are submitted in [Logstash](http://logstash.net/docs/1.4.2/tutorials/getting-started-with-logstash)
format (in JSON) to an [Elasticsearch](http://www.elasticsearch.org/) cluster which runs as a Kubernetes service.
As of Kubernetes 0.11, when you create a cluster the console output reports the URL of both the Elasticsearch cluster as well as
a URL for a [Kibana](http://www.elasticsearch.org/overview/kibana/) dashboard viewer for the logs that have been ingested
into Elasticsearch.
```
Cluster logs are ingested into Elasticsearch running at https://130.211.152.93/api/v1beta1/proxy/services/elasticsearch-logging/
Kibana logging dashboard will be available at https://130.211.152.93/api/v1beta1/proxy/services/kibana-logging/ (note the trailing slash)
```
Visiting the Kibana dashboard URL in a browser should give a display like this:
![Kibana](kibana.png)

To learn how to query, filter etc. using Kibana you might like to look at this [tutorial](http://www.elasticsearch.org/guide/en/kibana/current/working-with-queries-and-filters.html).

You can check to see if any logs are being ingested into Elasticsearch by curling against its URL. You will need to provide the username and password that was generated when your cluster was created. This can be found in the `kubernetes_auth` file for your cluster.
```
curl -k -u admin:Drt3KdRGnoQL6TQM https://130.211.152.93/api/v1beta1/proxy/services/elasticsearch-logging/_search?size=10
{"took":7,"timed_out":false,"_shards":{"total":5,"successful":5,"failed":0},"hits":{"total":3705,"max_score":1.0,"hits":[{"_index":"logstash-2015.01.08","_type":"fluentd","_id":"AUrK0hRRanI4L8Durpdh","_score":1.0,"_source":{"message":"I0108 18:30:47.694725    4927 server.go:313] GET /healthz: (9.249us) 200","tag":"kubelet","@timestamp":"2015-01-08T18:30:47+00:00"}},{"_index":"logstash-2015.01.08","_type":"fluentd","_id":"AUrK0hRRanI4L8Durpdm","_score":1.0,"_source":{"message":"E0108 18:30:52.299372    4927 metadata.go:109] while reading 'google-dockercfg' metadata: http status code: 404 while fetching url http://metadata.google.internal./computeMetadata/v1/instance/attributes/google-dockercfg","tag":"kubelet","@timestamp":"2015-01-08T18:30:52+00:00"}},{"_index":"logstash-2015.01.08","_type":"fluentd","_id":"AUrK0hRRanI4L8Durpdr","_score":1.0,"_source":{"message":"I0108 18:30:52.317636    4927 docker.go:214] Pulling image kubernetes/kube2sky without credentials","tag":"kubelet","@timestamp":"2015-01-08T18:30:52+00:00"}},{"_index":"logstash-2015.01.08","_type":"fluentd","_id":"AUrK0hRRanI4L8Durpdw","_score":1.0,"_source":{"message":"I0108 18:30:54.500174    4927 event.go:92] Event(api.ObjectReference{Kind:\"Pod\", Namespace:\"default\", Name:\"67cfcb1f-9764-11e4-898c-42010af03582\", UID:\"67cfcb1f-9764-11e4-898c-42010af03582\", APIVersion:\"v1beta1\", ResourceVersion:\"\", FieldPath:\"spec.containers{kube2sky}\"}): status: 'waiting', reason: 'created' Created with docker id ff24ec6eb3b10d1163a2bcb7c63ccef78e6e3e7a1185eba3fe430f6b3d871eb5","tag":"kubelet","@timestamp":"2015-01-08T18:30:54+00:00"}},{"_index":"logstash-2015.01.08","_type":"fluentd","_id":"AUrK0hRRanI4L8Durpd1","_score":1.0,"_source":{"message":"goroutine 114 [running]:","tag":"kubelet","@timestamp":"2015-01-08T18:30:56+00:00"}},{"_index":"logstash-2015.01.08","_type":"fluentd","_id":"AUrK0hRRanI4L8Durpd6","_score":1.0,"_source":{"message":"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet.(*Server).error(0xc2080e0060, 0x7fe0ba496840, 0xc208278840, 0x7fe0ba4881b0, 0xc20822daa0)","tag":"kubelet","@timestamp":"2015-01-08T18:30:56+00:00"}},{"_index":"logstash-2015.01.08","_type":"fluentd","_id":"AUrK0hRRanI4L8DurpeB","_score":1.0,"_source":{"message":"\t/go/src/github.com/GoogleCloudPlatform/kubernetes/_output/dockerized/go/src/github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/server.go:94 +0x44","tag":"kubelet","@timestamp":"2015-01-08T18:30:56+00:00"}},{"_index":"logstash-2015.01.08","_type":"fluentd","_id":"AUrK0hRSanI4L8DurpeJ","_score":1.0,"_source":{"message":"goroutine 114 [running]:","tag":"kubelet","@timestamp":"2015-01-08T18:30:56+00:00"}},{"_index":"logstash-2015.01.08","_type":"fluentd","_id":"AUrK0hRSanI4L8DurpeO","_score":1.0,"_source":{"message":"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet.(*Server).error(0xc2080e0060, 0x7fe0ba496840, 0xc208278a80, 0x7fe0ba4881b0, 0xc20822df00)","tag":"kubelet","@timestamp":"2015-01-08T18:30:56+00:00"}},{"_index":"logstash-2015.01.08","_type":"fluentd","_id":"AUrK0hRSanI4L8DurpeT","_score":1.0,"_source":{"message":"\t/go/src/github.com/GoogleCloudPlatform/kubernetes/_output/dockerized/go/src/github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/server.go:240 +0x45","tag":"kubelet","@timestamp":"2015-01-08T18:30:56+00:00"}}]}}
```
A [demonstration](../cluster/addons/fluentd-elasticsearch/logging-demo/README.md) of two synthetic logging sources can be used
to check that logging is working correctly.

Cluster logging can be turned on or off using the environment variable `ENABLE_NODE_LOGGING` which is defined in the
`config-default.sh` file for each provider. For the GCE provider this is set by default to `true`. Set this
to `false` to disable cluster logging.

The type of logging is used is specified by the environment variable `LOGGING_DESTINATION` which for the
GCE provider has the default value `elasticsearch`. If this is set to `gcp` for the GCE provider then
logs will be sent to the Google Cloud Logging system instead.

When using Elasticsearch the number of Elasticsearch instances can be controlled by setting the
variable `ELASTICSEARCH_LOGGING_REPLICAS` which has the default value of `1`. For large clusters
or clusters that are generating log information at a high rate you may wish to use more
Elasticsearch instances.

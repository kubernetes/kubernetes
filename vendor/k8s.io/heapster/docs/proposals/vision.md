# Heapster Long Term Vision

## Current status

Heapster is an important component of Kubernetes that is responsible for metrics and event 
handling. It reads metrics from cluster nodes and writes them to external, permanent 
storage. This is the main use case of Heapster.

To support system components of Kubernetes Heapster calculates aggregated metrics (like 
sum of containers' CPU usage in a pod) and long term statistics (average, 95percentile 
with 1h resolution), keeps them in memory and exposes via Heapster API. This API is mainly
used by Horizontal Pod Autoscaler which asks for the most recent performance related 
metrics to adjust the number of pods to the incoming traffic. The API is also used by KubeDash
and will be used by the new UI (which will replace KubeDash) as well. 

Additionally Heapster API allows to list all active nodes, namespaces, pods, containers 
etc. present in the system.

There is also a HeapsterGKE API dedicated for GKE through which it’s possible to get a full 
dump of all metrics (spanning last minute or two).

Metrics are gathered from cluster nodes, but Heapster developers wanted it to be useful also 
in non-Kubernetes clusters. They wrote Heapster in a such a way that metrics can be read not 
only from Kubernetes nodes (via Kubelet API) but also from custom deployments via cAdvisor 
(with support for CoreOS Fleet and flat file node lists). 

Metrics collected by Heapster can be written into multiple kinds of storage - Influxdb, 
OpenTSDB, Google Cloud Monitoring, Hawkular, Kafka, Riemann, ElasticSearch (some of them are
not yet submitted).

In addition to gathering metrics  Heapster is responsible for handling Kubernetes events - it 
reads them from Kubernetes API server and writes them, without extra processing, to a selection
of persistent storages: Google Cloud Logging, Influxdb, Kafka, OpenTSDB, Hawkular, 
ElasticSearch, etc.

There is/was a plan to add resource prediction components (Initial Resources, Vertical 
Pod Autoscaling) to Heapster binary.

## Separation of Use Cases
From the current state description (see above) the following use cases can be extracted:

* [UC1] Read metrics from nodes and write them to an external storage.
* [UC2] Expose metrics from the last 2-3 minutes (for HPA and GKE)
* [UC3] Read Events from the API server and write them to a permanent storage
* [UC4] Do some long-term (hours, days) metrics analysis to get stats (average, 95 percentile) 
and expected resource usage.
* [UC5] Provide cpu and memory metrics for longer time window for the new Kubernetes 
Dashboard UI (15 min for 1.2, up to 1h later for plots) 

UC1 and UC2 go together - to expose the most recent metrics the API should be connected 
to the metrics stream.
UC3 can be completely separated from UC1, UC2 and UC4 - it reads different data from a 
different place and writes it in a slightly different format to different sinks. 
UC4 is connected to UC1 and UC2 but it is more based on data from the permanent storage 
than on the super-fresh metrics stored in the memory.
UC5 can go either with UC1/UC2 or with UC4. As there is no immediate need for UC4 we will
 provide basic UC5 together with UC1/UC2 but in the future it will join UC4.

This separation leads to an idea of splitting Heapster into 3 binaries:

* Core Heapster - covering UC1, UC2 and temporarily UC5
* Eventer - covering UC3
* Oldtimer - covering UC4 and UC5 

## Reduction of Responsibility

With 3 possible node sources (Kuberentes API Server, flat file, CoreOS Fleet), 2 metrics 
sources (cAdvisor and Kubelet) and constantly growing number of sinks we have to separate 
the stuff that the core Heapster/K8S team is responsible for and what is provided as a 
plugin/addition and doesn’t come in the main release package. 

We decided to focus only on:

Kubernetes API Server node source
Kubelet metrics source
Influxdb, GCM, GKE (there is special endpoint for GKE that exposes all available metrics),
 Hawkular sinks for Heapster
Influxdb, GCL sinks for Eventer

The rest of the sources/sinks will be available as plugins. The plugin will be used in 2 flavors:

* Complied in - will require the user to rebuild the package and create his own image with 
the desired set of plugins.  
* Side-car - Heapster will talk to plugin’s HTTP server to get/pass metrics through a well 
defined json interface. The plugin runs in a separate container.

K8s team will explicitly say that it is NOT giving any warranty on the plugins. Plugins e2e 
tests can be included in some CI suite but we will not block our development (too much) if 
something breaks Kafka or Riemann. We will also not pay attention to whether a particular sink scales up.

For now we will keep all of the currently available sinks compiled-in by default, to keep the 
new Heapster more or less compatible with the old one, but eventually (if the number of sinks grows)
 we will migrate some of them to plugins.

## Custom Metrics Status

Heapster is not a generic solution for gathering arbitrary number of arbitrary-formated custom 
metrics. The support for custom metrics is focused on auto-scaling and critical functionality 
monitoring (and potentially scheduling). And Heapster is oriented towards system metrics, not 
application/business level metrics.

Kubernetes users and application developers will be able to push any number of their custom 
metrics through our pipeline to the storage but this should be considered as a bonus/best effort 
functionality. Custom metrics will not influence our performance targets (no extra fine-tunning effort 
to support >5 custom metrics per pod). There will be a flag in Kubelet that will limit the 
number of custom metrics.

## Performance Target

Heapster product family (Core, Eventer and Oldtimer) should follow the same performance goals 
as core Kubernetes. As Eventer is fairly simple and Oldtimer not yet fully defined this section
will focus only on Core Heapster (for metrics).

For 1.2 we should scale to 1000 nodes each running at least 30 pods (100 for 1.3) each reporting 
20 metrics every 1 min (30 sec preferably). That brings us to the number of  600k metrics 
per minute and 10k metrics per second.

Stretch goal (for 1.2/1.3) is 60k metrics per second (possibly with not everything being written to Influxdb). 
On smaller deployments, like 500 nodes with 15-30 pods each it should be easy to have 30 sec 
metrics resolution or smaller. 

Memory target - Fit into 2 gb with 1000 nodes x 30 pods and 6 gb with 1000 node x 100 pods (~60kb per pod). 

Latency, measured from the time when we initiate scraping metrics to the moment the metric 
change is visible in the API, should be less than 1*metrics resolution, which mainly depends 
on how fast it is possible to get all the metrics through the wire and parse them. 

The e2e latency from the moment the metric changes in the container to the moment the change is 
visible in Heapster API is: metric_resolution + heapster_latency.


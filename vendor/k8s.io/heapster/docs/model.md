# Heapster Metric Model

The Heapster Model is a structured representation of metrics for Kubernetes clusters, which is exposed through a set of REST API endpoints.
It allows the extraction of up to 15 minutes of historical data for any Container, Pod, Node or Namespace in the cluster, as well as the cluster itself (depending on the metric).

**Please bear in mind that this is not an official Kubernetes API, we will try to keep it stable but we don't guarantee that we won't change it in the future.**

## Usage

The Heapster Model is enabled by default. The resolution of the model can be configured through
the `-model_resolution` flag, which will cause the model to store historical data at the specified resolution. If the `-model_resolution` flag is not specified, the default resolution of 30 seconds will be used.

## API documentation

A detailed documentation of each API endpoint is listed below. 

All endpoints ending in `/metrics/{metric-name}/` can accept the optional `start` and `end` query parameters 
that represent the start and end time of the requested timeseries. The result
will be a list of (Timestamp, Value) pairs in the time range [start, end].
`start` and `end` are strings formatted according to RFC3339. If `start` is not
defined, it is assumed as the zero Unix epoch time. If `end` is not defined,
then all data later than `start` will be returned.

### Cluster-level Metrics

`/api/v1/model/metrics/`: Returns a list of available cluster-level metrics.

`/api/v1/model/metrics/{metric-name}?start=X&end=Y`: Returns a set of (Timestamp, Value) 
pairs for the requested cluster-level metric, between the time range specified by `start` and `end`. 

### Node-level Metrics
`/api/v1/model/nodes/`: Returns a list of all available nodes.

`/api/v1/model/nodes/{node-name}/metrics/`: Returns a list of available
node-level metrics.

`/api/v1/model/nodes/{node-name}/metrics/{metric-name}?start=X&end=Y`: Returns a set of (Timestamp, Value) 
pairs for the requested node-level metric, within the time range specified by `start` and `end`. 

### Namespace-level Metrics 
`/api/v1/model/namespaces/`: Returns a list of all available namespaces.

`/api/v1/model/namespaces/{namespace-name}/metrics/`: Returns a list of available namespace-level metrics.

`/api/v1/model/namespaces/{namespace-name}/metrics/{metric-name}?start=X&end=Y`: Returns a set of (Timestamp, Value) 
pairs for the requested namespace-level metric, within the time range specified by `start` and `end`. 


### Pod-level Metrics
`/api/v1/model/namespaces/{namespace-name}/pods/`: Returns a list of all available pods under a given namespace.

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/metrics/`: Returns a list of available pod-level metrics

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/metrics/{metric-name}?start=X&end=Y`: Returns a set of (Timestamp, Value) 
pairs for the requested pod-level metric, within the time range specified by `start` and `end`. 

### Container-level Metrics
Container metrics and stats are accessible for both containers that belong to
pods, as well as for free containers running in each node.

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/containers/`: Returns a list of all available containers under a given pod.

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/containers/{container-name}/metrics/`: Returns a list of available container-level metrics

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/containers/{container-name}/metrics/{metric-name}?start=X&end=Y`: Returns a set of (Timestamp, Value) 
pairs for the requested container-level metric, within the time range specified by `start` and `end`. 

`/api/v1/model/nodes/{node-name}/freecontainers/`: Returns a list of all available free containers under a given node.

`/api/v1/model/nodes/{node-name}/freecontainers/{container-name}/metrics/`: Returns a list of available container-level metrics

`/api/v1/model/nodes/{node-name}/freecontainers/{container-name}/metrics/{metric-name}?start=X&end=Y`: Returns a set of (Timestamp, Value) 
pairs for the requested container-level metric, within the time range specified by `start` and `end`. 

### Metric Types

All metrics available in the [storage schema](storage-schema.md) are also available through the api.

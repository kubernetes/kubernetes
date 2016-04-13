# Heapster Metric Model

The Heapster Model is a structured representation of metrics for Kubernetes clusters, which is exposed through a set of REST API endpoints.
It allows the extraction of up to 1 hour of historical data for any Container, Pod, Node or Namespace in the cluster, as well as the cluster itself.
The model can also export the Average, Max and 95th Percentile for each one of these entities, over a duration of 1 minute, 1 hour or 24 hours.



## Usage

The Heapster Model can be enabled by initializing Heapster with the `-use_model=true` flag. The resolution of the model can be configured through
the `-model_resolution` flag, which will cause the model to store historical data at the specified resolution. If the `-model_resolution` flag is not specified, the default
resolution of 1 minute will be used.

## API documentation

A detailed documentation of each API endpoint is listed below. 

All endpoints ending in `/metrics/{metric-name}/` can accept the optional `start` and `end` query parameters 
that represent the start and end time of the requested timeseries. The result
will be a list of (Timestamp, Value) pairs in the time range [start, end].
`start` and `end` are strings formatted according to RFC3339. If `start` is not
defined, it is assumed as the zero Unix epoch time. If `end` is not defined,
then all data later than `start` will be returned.

### Cluster-level Metrics and Stats
`/api/v1/model/`: The root path of the model API, displays all browsable subpaths.

`/api/v1/model/metrics/`: Returns a list of available cluster-level metrics.

`/api/v1/model/metrics/{metric-name}?start=X&end=Y`: Returns a set of (Timestamp, Value) 
pairs for the requested cluster-level metric, between the time range specified by `start` and `end`. 

`/api/v1/model/stats/`: Exposes the average, max and 95th percentile over the
past minute, hour and day for each cluster metric.

### Node-level Metrics and Stats
`/api/v1/model/nodes/`: Returns a list of all available nodes, along
with their latest CPU and Memory Usage values.

`/api/v1/model/nodes/{node-name}/`: Returns all browsable subpaths for a
specific node.

`/api/v1/model/nodes/{node-name}/pods/`: Returns a list of all available pods
under a given node, along with their latest CPU and Memory Usage values.

`/api/v1/model/nodes/{node-name}/metrics/`: Returns a list of available
node-level metrics.

`/api/v1/model/nodes/{node-name}/metrics/{metric-name}?start=X&end=Y`: Returns a set of (Timestamp, Value) 
pairs for the requested node-level metric, within the time range specified by `start` and `end`. 

`/api/v1/model/nodes/{node-name}/stats/`: Exposes the average, max and 95th
percentile over the past minute, hour and day for each node metric.


### Namespace-level Metrics and Stats
`/api/v1/model/namespaces/`: Returns a list of all available namespaces, along
with their latest CPU and Memory Usage values.

`/api/v1/model/namespaces/{namespace-name}/`: Returns all browsable subpaths for
a specific namespace.

`/api/v1/model/namespaces/{namespace-name}/metrics/`: Returns a list of available namespace-level metrics.

`/api/v1/model/namespaces/{namespace-name}/metrics/{metric-name}?start=X&end=Y`: Returns a set of (Timestamp, Value) 
pairs for the requested namespace-level metric, within the time range specified by `start` and `end`. 

`/api/v1/model/namespaces/{namespace-name}/stats/`: Exposes the average, max and 95th percentile over the
past minute, hour and day for each namespace metric.


### Pod-level Metrics and Stats
`/api/v1/model/namespaces/{namespace-name}/pods/`: Returns a list of all available pods under a given namespace, along
with their latest CPU and Memory Usage values.

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/`: Returns all browsable subpaths for
a specific Pod.

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/metrics/`: Returns a list of available pod-level metrics

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/metrics/{metric-name}?start=X&end=Y`: Returns a set of (Timestamp, Value) 
pairs for the requested pod-level metric, within the time range specified by `start` and `end`. 

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/stats/`: Exposes the average, max and 95th percentile over the
past minute, hour and day for each pod-level metric.


### Container-level Metrics and Stats
Container metrics and stats are accessible for both containers that belong to
pods, as well as for free containers running in each node.

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/containers/`: Returns a list of all available containers under a given pod, along
with their latest CPU and Memory Usage values.

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/containers/{container-name}/`: Returns all browsable subpaths for
a specific container.

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/containers/{container-name}/metrics/`: Returns a list of available container-level metrics

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/containers/{container-name}/metrics/{metric-name}?start=X&end=Y`: Returns a set of (Timestamp, Value) 
pairs for the requested container-level metric, within the time range specified by `start` and `end`. 

`/api/v1/model/namespaces/{namespace-name}/pods/{pod-name}/containers/{container-name}/stats/`: Exposes the average, max and 95th percentile over the
past minute, hour and day for each container metric.

`/api/v1/model/nodes/{node-name}/freecontainers/`: Returns a list of all available free containers under a given node, along
with their latest CPU and Memory Usage values.

`/api/v1/model/nodes/{node-name}/freecontainers/{container-name}/`: Returns all browsable subpaths for
a specific free container.

`/api/v1/model/nodes/{node-name}/freecontainers/{container-name}/metrics/`: Returns a list of available container-level metrics

`/api/v1/model/nodes/{node-name}/freecontainers/{container-name}/metrics/{metric-name}?start=X&end=Y`: Returns a set of (Timestamp, Value) 
pairs for the requested container-level metric, within the time range specified by `start` and `end`. 

`/api/v1/model/nodes/{node-name}/freecontainers/{container-name}/stats/`: Exposes the average, max and 95th percentile over the
past minute, hour and day for each container metric.

### Metric Types

* cpu-limit
* cpu-usage
* memory-limit
* memory-usage
* memory-working
* fs-limit-<fs_name>
* fs-usage-<fs_name>

### Sample API response

	curl http://heapster:8082/api/v1/model/stats/

```json
{
  "uptime": 2543160,
  "stats": {
   "cpu-limit": {
    "minute": {
     "average": 1000,
     "percentile": 1000,
     "max": 1000
    },
    "hour": {
     "average": 1000,
     "percentile": 1000,
     "max": 1000
    },
    "day": {
     "average": 1000,
     "percentile": 1000,
     "max": 1000
    }
   },
   "cpu-usage": {
    "minute": {
     "average": 10,
     "percentile": 10,
     "max": 10
    },
    "hour": {
     "average": 10,
     "percentile": 10,
     "max": 10
    },
    "day": {
     "average": 10,
     "percentile": 10,
     "max": 10
    }
   },
   "fs-limit-dev-disk-by-uuid-dcaa07b0-d2ad-4a32-bd61-6584d0da68c0": {
    "minute": {
     "average": 21103243264,
     "percentile": 21103243264,
     "max": 21103243264
    },
    "hour": {
     "average": 21103243300,
     "percentile": 21103243300,
     "max": 21103243300
    },
    "day": {
     "average": 21103243300,
     "percentile": 21103243300,
     "max": 21103243300
    }
   },
   "fs-usage-dev-disk-by-uuid-dcaa07b0-d2ad-4a32-bd61-6584d0da68c0": {
    "minute": {
     "average": 12974346240,
     "percentile": 12974346240,
     "max": 12974346240
    },
    "hour": {
     "average": 12974346300,
     "percentile": 12974346300,
     "max": 12974346300
    },
    "day": {
     "average": 12974346300,
     "percentile": 12974346300,
     "max": 12974346300
    }
   },
   "memory-limit": {
    "minute": {
     "average": 0,
     "percentile": 18446744073709551615,
     "max": 18446744073709551615
    },
    "hour": {
     "average": 1366425486941603612,
     "percentile": 12297829382474432512,
     "max": 18446744073709551615
    },
    "day": {
     "average": 1366425486941603612,
     "percentile": 12297829382474432512,
     "max": 18446744073709551615
    }
   },
   "memory-usage": {
    "minute": {
     "average": 1824296960,
     "percentile": 1824296960,
     "max": 1824296960
    },
    "hour": {
     "average": 1820327936,
     "percentile": 1820327936,
     "max": 1824296960
    },
    "day": {
     "average": 1820327936,
     "percentile": 1820327936,
     "max": 1824296960
    }
   },
   "memory-working": {
    "minute": {
     "average": 447021056,
     "percentile": 447021056,
     "max": 447021056
    },
    "hour": {
     "average": 444596224,
     "percentile": 444596224,
     "max": 447021056
    },
    "day": {
     "average": 444596224,
     "percentile": 444596224,
     "max": 447021056
    }
   }
  }
 }

```

# cAdvisor Remote REST API

cAdvisor exposes its raw and processed stats via a versioned remote REST API:

`http://<hostname>:<port>/api/<version>/<request>`

This document covers the detail of version 2.0. All resources covered in this version are read-only.

NOTE: v2.0 is still a work in progress.

## Version information

Software version for cAdvisor can be obtained from version endpoint as follows:
`/api/v2.0/version`

## Machine Information

The resource name for machine information is as follows:

`/api/v2.0/machine`

The machine information is returned as a JSON object of the `MachineInfo` struct found in [info/v1/machine.go](../info/v1/machine.go)

## Attributes

Attributes endpoint provides hardware and software attributes of the running machine.
The resource name for attributes is:
`/api/v2.0/attributes`

Hardware information includes all information covered by machine endpoint. Software information include version of cAdvisor, kernel, docker, and underlying OS.

The actual object is the marshalled JSON of the `Attributes` struct found in [info/v2/machine.go](../info/v2/machine.go)

## Container Stats
The resource name for container stats information is:
`/api/v2.0/stats/<container identifier>`

### Stats request options

Stats support following options in the request:
- `type`: describes the type of identifier. Supported values are `name`(default) and `docker`. `name` implies that the identifier is an absolute container name. `docker` implies that the identifier is a docker id.
- `recursive`: Option to specify if stats for subcontainers of the requested containers should also be reported. Default is false.
- `count`: Number of stats samples to be reported. Default is 64.

### Container name

When container identifier is of type `name`, the identifier is interpreted as the absolute container name. Naming follows the lmctfy convention. For example:

| Container Name       | Resource Name                             |
|----------------------|-------------------------------------------|
| /                    | /api/v2.0/containers/                     |
| /foo                 | /api/v2.0/containers/foo                  |
| /docker/2c4dee605d22 | /api/v2.0/containers/docker/2c4dee605d22  |

Note that the root container (`/`) contains usage for the entire machine. All Docker containers are listed under `/docker`. Also, `type=name` is not required in the examples above as `name` is the default type.

### Docker Containers

When container identifier is of type `docker`, the identifier is interpreted as docker id. For example:


| Docker container     | Resource Name                             |
|----------------------|-------------------------------------------|
| All docker containers| /api/v2.0/stats?type=docker&recursive=true|
| clever_colden        | /api/v2.0/stats/clever_colden?type=docker |
| 2c4dee605d22         | /api/v2.0/stats/2c4dee605d22?type=docker  |

The Docker name can be either the UUID or the short name of the container. It returns the information of the specified container(s).

Note that `recursive` is only valid when docker root is specified. It is used to get stats for all docker containers.

### Returned stats

The stats information is returned  as a JSON object containing a map from container name to list of stat objects. Stat object is the marshalled JSON of the `ContainerStats` struct found in [info/v2/container.go](../info/v2/container.go)

## Container Stats Summary
Instead of a list of periodically collected detailed samples, cAdvisor can also provide a summary of stats for a container. It provides the latest collected stats and percentiles (max, average, and 90%ile) values for usage in last minute and hour. (Usage summary for last day exists, but is not currently used.)

Unlike the regular stats API, only selected resources are captured by `summary`. Currently it is limited to cpu and memory usage.

The resource name for container summary information is:
`/api/v2.0/summary/<container identifier>`

Additionally, `type` and `recursive` options can be used to describe the identifier type and ask for summary of all subcontainers respectively. The semantics are same as described for container stats above.

The returned summary information is a JSON object containing a map from container name to list of summary objects. Summary object is the marshalled JSON of the `DerivedStats` struct found in [info/v2/container.go](../info/v2/container.go)

## Container Spec

The resource name for container stats information is:
`/api/v2.0/spec/<container identifier>`

Additionally, `type` and `recursive` options can be used to describe the identifier type and ask for spec of all subcontainers respectively. The semantics are same as described for container stats above.

The spec information is returned as a JSON object containing a map from container name to list of spec objects. Spec object is the marshalled JSON of the `ContainerSpec` struct found in [info/v2/container.go](../info/v2/container.go)


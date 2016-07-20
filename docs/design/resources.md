<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/design/resources.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
**Note: this is a design doc, which describes features that have not been
completely implemented. User documentation of the current state is
[here](../user-guide/compute-resources.md). The tracking issue for
implementation of this model is [#168](http://issue.k8s.io/168). Currently, both
limits and requests of memory and cpu on containers (not pods) are supported.
"memory" is in bytes and "cpu" is in milli-cores.**

# The Kubernetes resource model

To do good pod placement, Kubernetes needs to know how big pods are, as well as
the sizes of the nodes onto which they are being placed. The definition of "how
big" is given by the Kubernetes resource model &mdash; the subject of this
document.

The resource model aims to be:
* simple, for common cases;
* extensible, to accommodate future growth;
* regular, with few special cases; and
* precise, to avoid misunderstandings and promote pod portability.

## The resource model

A Kubernetes _resource_ is something that can be requested by, allocated to, or
consumed by a pod or container. Examples include memory (RAM), CPU, disk-time,
and network bandwidth.

Once resources on a node have been allocated to one pod, they should not be
allocated to another until that pod is removed or exits. This means that
Kubernetes schedulers should ensure that the sum of the resources allocated
(requested and granted) to its pods never exceeds the usable capacity of the
node. Testing whether a pod will fit on a node is called _feasibility checking_.

Note that the resource model currently prohibits over-committing resources; we
will want to relax that restriction later.

### Resource types

All resources have a _type_ that is identified by their _typename_ (a string,
e.g., "memory").  Several resource types are predefined by Kubernetes (a full
list is below), although only two will be supported at first: CPU and memory.
Users and system administrators can define their own resource types if they wish
(e.g., Hadoop slots).

A fully-qualified resource typename is constructed from a DNS-style _subdomain_,
followed by a slash `/`, followed by a name.
* The subdomain must conform to [RFC 1123](http://www.ietf.org/rfc/rfc1123.txt)
(e.g., `kubernetes.io`, `example.com`).
* The name must be not more than 63 characters, consisting of upper- or
lower-case alphanumeric characters, with the `-`, `_`, and `.` characters
allowed anywhere except the first or last character.
* As a shorthand, any resource typename that does not start with a subdomain and
a slash will automatically be prefixed with the built-in Kubernetes _namespace_,
`kubernetes.io/` in order to fully-qualify it. This namespace is reserved for
code in the open source Kubernetes repository; as a result, all user typenames
MUST be fully qualified, and cannot be created in this namespace.

Some example typenames include `memory` (which will be fully-qualified as
`kubernetes.io/memory`), and `example.com/Shiny_New-Resource.Type`.

For future reference, note that some resources, such as CPU and network
bandwidth, are _compressible_, which means that their usage can potentially be
throttled in a relatively benign manner. All other resources are
_incompressible_, which means that any attempt to throttle them is likely to
cause grief. This distinction will be important if a Kubernetes implementation
supports over-committing of resources.

### Resource quantities

Initially, all Kubernetes resource types are _quantitative_, and have an
associated _unit_ for quantities of the associated resource (e.g., bytes for
memory, bytes per seconds for bandwidth, instances for software licences). The
units will always be a resource type's natural base units (e.g., bytes, not MB),
to avoid confusion between binary and decimal multipliers and the underlying
unit multiplier (e.g., is memory measured in MiB, MB, or GB?).

Resource quantities can be added and subtracted: for example, a node has a fixed
quantity of each resource type that can be allocated to pods/containers; once
such an allocation has been made, the allocated resources cannot be made
available to other pods/containers without over-committing the resources.

To make life easier for people, quantities can be represented externally as
unadorned integers, or as fixed-point integers with one of these SI suffices
(E, P, T, G, M, K, m) or their power-of-two equivalents (Ei, Pi, Ti, Gi, Mi,
 Ki). For example, the following represent roughly the same value: 128974848,
"129e6", "129M" , "123Mi".  Small quantities can be represented directly as
decimals (e.g., 0.3), or using milli-units (e.g., "300m").
  * "Externally" means in user interfaces, reports, graphs, and in JSON or YAML
resource specifications that might be generated or read by people.
  * Case is significant: "m" and "M" are not the same, so "k" is not a valid SI
suffix. There are no power-of-two equivalents for SI suffixes that represent
multipliers less than 1.
  * These conventions only apply to resource quantities, not arbitrary values.

Internally (i.e., everywhere else), Kubernetes will represent resource
quantities as integers so it can avoid problems with rounding errors, and will
not use strings to represent numeric values. To achieve this, quantities that
naturally have fractional parts (e.g., CPU seconds/second) will be scaled to
integral numbers of milli-units (e.g., milli-CPUs) as soon as they are read in.
Internal APIs, data structures, and protobufs will use these scaled integer
units. Raw measurement data such as usage may still need to be tracked and
calculated using floating point values, but internally they should be rescaled
to avoid some values being in milli-units and some not.
  * Note that reading in a resource quantity and writing it out again may change
the way its values are represented, and truncate precision (e.g., 1.0001 may
become 1.000), so comparison and difference operations (e.g., by an updater)
must be done on the internal representations.
  * Avoiding milli-units in external representations has advantages for people
who will use Kubernetes, but runs the risk of developers forgetting to rescale
or accidentally using floating-point representations. That seems like the right
choice. We will try to reduce the risk by providing libraries that automatically
do the quantization for JSON/YAML inputs.

### Resource specifications

Both users and a number of system components, such as schedulers, (horizontal)
auto-scalers, (vertical) auto-sizers, load balancers, and worker-pool managers
need to reason about resource requirements of workloads, resource capacities of
nodes, and resource usage. Kubernetes divides specifications of *desired state*,
aka the Spec, and representations of *current state*, aka the Status. Resource
requirements and total node capacity fall into the specification category, while
resource usage, characterizations derived from usage (e.g., maximum usage,
histograms), and other resource demand signals (e.g., CPU load) clearly fall
into the status category and are discussed in the Appendix for now.

Resource requirements for a container or pod should have the following form:

```yaml
resourceRequirementSpec: [
  request:   [ cpu: 2.5, memory: "40Mi" ],
  limit:     [ cpu: 4.0, memory: "99Mi" ],
]
```

Where:
* _request_ [optional]: the amount of resources being requested, or that were
requested and have been allocated. Scheduler algorithms will use these
quantities to test feasibility (whether a pod will fit onto a node).
If a container (or pod) tries to use more resources than its _request_, any
associated SLOs are voided &mdash; e.g., the program it is running may be
throttled (compressible resource types), or the attempt may be denied. If
_request_ is omitted for a container, it defaults to _limit_ if that is
explicitly specified, otherwise to an implementation-defined value; this will
always be 0 for a user-defined resource type. If _request_ is omitted for a pod,
it defaults to the sum of the (explicit or implicit) _request_ values for the
containers it encloses.

* _limit_ [optional]: an upper bound or cap on the maximum amount of resources
that will be made available to a container or pod; if a container or pod uses
more resources than its _limit_, it may be terminated. The _limit_ defaults to
"unbounded"; in practice, this probably means the capacity of an enclosing
container, pod, or node, but may result in non-deterministic behavior,
especially for memory.

Total capacity for a node should have a similar structure:

```yaml
resourceCapacitySpec: [
  total:     [ cpu: 12,  memory: "128Gi" ]
]
```

Where:
* _total_: the total allocatable resources of a node. Initially, the resources
at a given scope will bound the resources of the sum of inner scopes.

#### Notes

  * It is an error to specify the same resource type more than once in each
list.

  * It is an error for the _request_ or _limit_ values for a pod to be less than
the sum of the (explicit or defaulted) values for the containers it encloses.
(We may relax this later.)

  * If multiple pods are running on the same node and attempting to use more
resources than they have requested, the result is implementation-defined. For
example: unallocated or unused resources might be spread equally across
claimants, or the assignment might be weighted by the size of the original
request, or as a function of limits, or priority, or the phase of the moon,
perhaps modulated by the direction of the tide. Thus, although it's not
mandatory to provide a _request_, it's probably a good idea. (Note that the
_request_ could be filled in by an automated system that is observing actual
usage and/or historical data.)

  * Internally, the Kubernetes master can decide the defaulting behavior and the
kubelet implementation may expected an absolute specification. For example, if
the master decided that "the default is unbounded" it would pass 2^64 to the
kubelet.


## Kubernetes-defined resource types

The following resource types are predefined ("reserved") by Kubernetes in the
`kubernetes.io` namespace, and so cannot be used for user-defined resources.
Note that the syntax of all resource types in the resource spec is deliberately
similar, but some resource types (e.g., CPU) may receive significantly more
support than simply tracking quantities in the schedulers and/or the Kubelet.

### Processor cycles

  * Name: `cpu` (or `kubernetes.io/cpu`)
  * Units: Kubernetes Compute Unit seconds/second (i.e., CPU cores normalized to
a canonical "Kubernetes CPU")
  * Internal representation: milli-KCUs
  * Compressible? yes
  * Qualities: this is a placeholder for the kind of thing that may be supported
in the future &mdash; see [#147](http://issue.k8s.io/147)
    * [future] `schedulingLatency`: as per lmctfy
    * [future] `cpuConversionFactor`: property of a node: the speed of a CPU
core on the node's processor divided by the speed of the canonical Kubernetes
CPU (a floating point value; default = 1.0).

To reduce performance portability problems for pods, and to avoid worse-case
provisioning behavior, the units of CPU will be normalized to a canonical
"Kubernetes Compute Unit" (KCU, pronounced ˈko͝oko͞o), which will roughly be
equivalent to a single CPU hyperthreaded core for some recent x86 processor. The
normalization may be implementation-defined, although some reasonable defaults
will be provided in the open-source Kubernetes code.

Note that requesting 2 KCU won't guarantee that precisely 2 physical cores will
be allocated &mdash; control of aspects like this will be handled by resource
_qualities_ (a future feature).


### Memory

  * Name: `memory` (or `kubernetes.io/memory`)
  * Units: bytes
  * Compressible? no (at least initially)

The precise meaning of what "memory" means is implementation dependent, but the
basic idea is to rely on the underlying `memcg` mechanisms, support, and
definitions.

Note that most people will want to use power-of-two suffixes (Mi, Gi) for memory
quantities rather than decimal ones: "64MiB" rather than "64MB".


## Resource metadata

A resource type may have an associated read-only ResourceType structure, that
contains metadata about the type. For example:

```yaml
resourceTypes: [
  "kubernetes.io/memory": [
    isCompressible: false, ... 
  ]
  "kubernetes.io/cpu": [
    isCompressible: true,
    internalScaleExponent: 3, ...
  ]
  "kubernetes.io/disk-space": [ ... ]
]
```

Kubernetes will provide ResourceType metadata for its predefined types. If no
resource metadata can be found for a resource type, Kubernetes will assume that
it is a quantified, incompressible resource that is not specified in
milli-units, and has no default value.

The defined properties are as follows:

| field name | type | contents |
| ---------- | ---- | -------- |
| name | string, required | the typename, as a fully-qualified string (e.g., `kubernetes.io/cpu`) |
| internalScaleExponent | int, default=0 | external values are multiplied by 10 to this power for internal storage (e.g., 3 for milli-units) |
| units | string, required | format: `unit* [per unit+]` (e.g., `second`, `byte per second`). An empty unit field means "dimensionless". |
| isCompressible | bool, default=false | true if the resource type is compressible |
| defaultRequest | string, default=none | in the same format as a user-supplied value |
| _[future]_ quantization | number, default=1 | smallest granularity of allocation: requests may be rounded up to a multiple of this unit; implementation-defined unit (e.g., the page size for RAM). |


# Appendix: future extensions

The following are planned future extensions to the resource model, included here
to encourage comments.

## Usage data

Because resource usage and related metrics change continuously, need to be
tracked over time (i.e., historically), can be characterized in a variety of
ways, and are fairly voluminous, we will not include usage in core API objects,
such as [Pods](../user-guide/pods.md) and Nodes, but will provide separate APIs
for accessing and managing that data. See the Appendix for possible
representations of usage data, but the representation we'll use is TBD.

Singleton values for observed and predicted future usage will rapidly prove
inadequate, so we will support the following structure for extended usage
information:

```yaml
resourceStatus: [
  usage:     [ cpu: <CPU-info>, memory: <memory-info> ],
  maxusage:  [ cpu: <CPU-info>, memory: <memory-info> ],
  predicted: [ cpu: <CPU-info>, memory: <memory-info> ],
]
```

where a `<CPU-info>` or `<memory-info>` structure looks like this:

```yaml
{
    mean: <value>    # arithmetic mean
    max: <value>     # minimum value
    min: <value>     # maximum value
    count: <value>   # number of data points
    percentiles: [   # map from %iles to values
      "10": <10th-percentile-value>,
      "50": <median-value>,
      "99": <99th-percentile-value>,
      "99.9": <99.9th-percentile-value>,
      ...
    ]
}
```

All parts of this structure are optional, although we strongly encourage
including quantities for 50, 90, 95, 99, 99.5, and 99.9 percentiles.
_[In practice, it will be important to include additional info such as the
length of the time window over which the averages are calculated, the
confidence level, and information-quality metrics such as the number of dropped
or discarded data points.]_ and predicted

## Future resource types

### _[future] Network bandwidth_

  * Name: "network-bandwidth" (or `kubernetes.io/network-bandwidth`)
  * Units: bytes per second
  * Compressible? yes

### _[future] Network operations_

  * Name: "network-iops" (or `kubernetes.io/network-iops`)
  * Units: operations (messages) per second
  * Compressible? yes

### _[future] Storage space_

  * Name: "storage-space" (or `kubernetes.io/storage-space`)
  * Units: bytes
  * Compressible? no

The amount of secondary storage space available to a container. The main target
is local disk drives and SSDs, although this could also be used to qualify
remotely-mounted volumes. Specifying whether a resource is a raw disk, an SSD, a
disk array, or a file system fronting any of these, is left for future work.

### _[future] Storage time_

  * Name: storage-time (or `kubernetes.io/storage-time`)
  * Units: seconds per second of disk time
  * Internal representation: milli-units
  * Compressible? yes

This is the amount of time a container spends accessing disk, including actuator
and transfer time. A standard disk drive provides 1.0 diskTime seconds per
second.

### _[future] Storage operations_

  * Name: "storage-iops" (or `kubernetes.io/storage-iops`)
  * Units: operations per second
  * Compressible? yes


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/resources.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

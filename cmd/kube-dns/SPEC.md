# Kubernetes DNS-Based Service Discovery

## About This Document

This document is a specification for DNS-based Kubernetes service
discovery. While service discovery in Kubernetes may be provided via
other protocols and mechanisms, DNS is very commonly used and is a
highly recommended add-on. The actual DNS service itself need not
be provided by the default Kube-DNS implementation. This document
is intended to provide a baseline for commonality between implementations.

## Schema Version

This document describes version 1.0.0 of the schema.

## Resource Records

Any DNS-based service discovery solution for Kubernetes must provide the
resource records (RRs) described below to be considered compliant with this
specification.

### Definitions

In the RR descriptions below, values not in angle brackets, `< >`,
are literals. The meaning of the values in angle brackets are defined
below or in the description of the specific record.

- `<zone>` = configured cluster domain, e.g. cluster.local
- `<ns>` = a Namespace
- `<ttl>` = the standard DNS time-to-live value for the record

In the RR descriptions below, the following definitions should be used for
words in _italics_.

_hostname_
  - In order of precedence, the _hostname_ of an endpoint is:
    - The value of the endpoint’s `hostname` field.
    - A unique, system-assigned identifier for the endpoint. The exact format and source of this identifier is not prescribed by this specification. However, it must be possible to use this to identify a specific endpoint in the context of a Service. This is used in the event no explicit endpoint hostname is defined.

_ready_
  - An endpoint is considered _ready_ if its address is in the `addresses` field of the EndpointSubset object, or the corresponding service has the `service.alpha.kubernetes.io/tolerate-unready-endpoints` annotation set to `true`.

All comparisons between query data and data in Kubernetes is **case-insensitive**.

### Record for Schema Version

There must be a `TXT` record named `dns-version.<zone>.` that contains the
[semantic version](http://semver.org) of the DNS schema in use in this cluster.

- Record Format:
  - `dns-version.<zone>. <ttl> IN TXT <schema-version>`
- Question Example:
  - `dns-version.cluster.local. IN TXT`
- Answer Example:
  - `dns-version.cluster.local. 28800 IN TXT "1.0.0"`

### Records for a Service with ClusterIP

Given a Service named `<service>` in Namespace `<ns>` with ClusterIP
`<cluster-ip>`, the following records must exist.

#### `A` Record
- Record Format:
  - `<service>.<ns>.svc.<zone>. <ttl> IN A <cluster-ip>`
- Question Example:
  - `kubernetes.default.svc.cluster.local. IN A`
- Answer Example:
  - `kubernetes.default.svc.cluster.local. 4 IN A 10.3.0.1`

#### `SRV` Records
For each port in the Service with name `<port>` and number
`<port-number>` using protocol `<proto>`, an `SRV` record of the following
form must exist.
- Record Format:
   - `_<port>._<proto>.<service>.<ns>.svc.<zone>. <ttl> IN SRV <weight> <priority> <port-number> <service>.<ns>.svc.<zone>.`

The priority `<priority>` and weight `<weight>` are numbers as described
in [RFC2782](https://tools.ietf.org/html/rfc2782) and whose values are not
prescribed by this specification.

Unnamed ports do not have an `SRV` record.

- Question Example:
  - `_https._tcp.kubernetes.default.svc.cluster.local. IN SRV`
- Answer Example:
  - `_https._tcp.kubernetes.default.svc.cluster.local. 30 IN SRV 10 100 443 kubernetes.default.svc.cluster.local.`

The Additional section of the response may include the Service `A` record
referred to in the `SRV` record.

#### `PTR` Record
Given Service ClusterIP `<a>.<b>.<c>.<d>`, a `PTR` record of the following
form must exist.
- Record Format:
  - `<d>.<c>.<b>.<a>.in-addr.arpa. <ttl> IN PTR <service>.<ns>.svc.<zone>.`
- Question Example:
  - `1.0.3.10.in-addr.arpa. IN PTR`
- Answer Example:
  - `1.0.3.10.in-addr.arpa. 14 IN PTR kubernetes.default.svc.cluster.local.`

### Records for a Headless Service

Given a headless Service `<service>` in Namespace `<ns>` (i.e., a Service with
no ClusterIP), the following records must exist.

#### `A` Records
There must be an `A` record for each _ready_ endpoint of the Service with IP
address `<endpoint-ip>` as shown below. If there are no _ready_ endpoints,
there must be no `A` records of this form; however, a query for them will have
an empty answer with `rcode` 0 rather than `NXDOMAIN`, since the Service exists.

- Record Format:
  - `<service>.<ns>.svc.<zone>. <ttl> IN A <endpoint-ip>`
- Question Example:
  - `headless.default.svc.cluster.local. IN A`
- Answer Example:
```
    headless.default.svc.cluster.local. 4 IN A 10.3.0.1
    headless.default.svc.cluster.local. 4 IN A 10.3.0.2
    headless.default.svc.cluster.local. 4 IN A 10.3.0.3
```

There must also be an `A` record of the following form for each _ready_
endpoint with _hostname_ of `<hostname>` and IP address `<endpoint-ip>`.
If there are multiple IP addresses for a given _hostname_, then there
must be one such `A` record returned for each IP.
- Record Format:
  - `<hostname>.<service>.<ns>.svc.<zone>. <ttl> IN A <endpoint-ip>`
- Question Example:
  - `my-pet.headless.default.svc.cluster.local. IN A`
- Answer Example:
  - `my-pet.headless.default.svc.cluster.local. 4 IN A 10.3.0.100`

#### `SRV` Records
For each combination of _ready_ endpoint with _hostname_ of `<hostname>`, and
port in the Service with name `<port>` and number `<port-number>` using
protocol `<proto>`, an `SRV` record of the following form must exist.
- Record Format:
   - `_<port>._<proto>.<service>.<ns>.svc.<zone>. <ttl> IN SRV <weight> <priority> <port-number> <hostname>.<service>.<ns>.svc.<zone>.`

This implies that if there are **N** _ready_ endpoints and the Service
defines **M** named ports, there will be **N** :heavy_multiplication_x: **M**
`SRV` RRs for the Service.

The priority `<priority>` and weight `<weight>` are numbers as described
in [RFC2782](https://tools.ietf.org/html/rfc2782) and whose values are not
prescribed by this specification.

Unnamed ports do not have an `SRV` record.

- Question Example:
  - `_https._tcp.headless.default.svc.cluster.local. IN SRV`
- Answer Example:
```
        _https._tcp.headless.default.svc.cluster.local. 4 IN SRV 10 100 443 my-pet.headless.default.svc.cluster.local.
        _https._tcp.headless.default.svc.cluster.local. 4 IN SRV 10 100 443 my-pet-2.headless.default.svc.cluster.local.
        _https._tcp.headless.default.svc.cluster.local. 4 IN SRV 10 100 443 438934893.headless.default.svc.cluster.local.
```

The Additional section of the response may include the `A` records
referred to in the `SRV` records.

#### `PTR` Records

Given a _ready_ endpoint with _hostname_ of `<hostname>` and IP address
`<a>.<b>.<c>.<d>`, a `PTR` record of the following form must exist.
- Record Format:
  - `<d>.<c>.<b>.<a>.in-addr.arpa. <ttl> IN PTR <hostname>.<service>.<ns>.svc.<zone>.`
- Question Example:
  - `100.0.3.10.in-addr.arpa. IN PTR`
- Answer Example:
  - `100.0.3.10.in-addr.arpa. 14 IN PTR my-pet.headless.default.svc.cluster.local.`

### Deprecated Records
Kube-DNS versions prior to implementation of this specification also replied
with an `A` record of the form below for any values of `<a>`, `<b>`, `<c>`, and `<d>` between 0 and 254:
- Record Format:
  - `<a>-<b>-<c>-<d>.<ns>.pod.<zone>. <ttl> IN A <a>.<b>.<c>.<d>`

This behavior is deprecated and not required to satisfy this specification.

## Schema Extensions
Specific implementations may choose to extend this schema, but the RRs in this
document must be a subset of the RRs produced by the implementation.

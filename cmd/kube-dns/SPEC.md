# Kubernetes DNS-Based Service Discovery

## About This Document

This document describes a DNS schema for DNS-based Kubernetes service
discovery. While service discovery in Kubernetes may be provided via
other protocols and mechanisms, DNS is very commonly used and is a
highly recommended add-on. The actual DNS service itself though need not
be provided by the default Kube-DNS implementation, and so this document
is intended to provide a baseline for commonality between
implementations.

## Schema Version

This document describes version 1.0.0 of the schema.

## Query Responses

Any DNS-based service discovery solution for Kubernetes must support the
queries described below to be considered compliant with this specification.

In the query names below, values not in angle brackets, `< >`,
are literals. The meaning of the values in angle brackets are defined
below or in the description of the specific query response.

`<zone>` = configured cluster domain, e.g. cluster.local

`<ns>` = a Namespace

All comparisons between query names and data in Kubernetes is **case-insensitive**.

In the results column, the following definitions should be used for
words in _italics_.

- _hostname_
  - For endpoints, in order of precedence:
    - The value of the endpoint’s `hostname` field.
    - A unique, system-assigned identifier for the endpoint. The exact format and source of this identifier is not prescribed by this specification. However, it must be possible to use this to identify a specific endpoint in the context of a Service. This is used for SRV queries below in the event no explicit endpoint hostname is defined by the above methods.

- _ready_
  - An endpoint is considered _ready_ if its address is in the `addresses` field of the EndpointSubset object, or the corresponding service has the `service.alpha.kubernetes.io/tolerate-unready-endpoints` annotation set to `true`.

The supported queries and expected responses follow.

- **Question:** `<service>.<ns>.svc.<zone>. IN A`
  - If no Service named `<service>` exists in `<ns>`, the result will be `NXDOMAIN`.
  - If a Service named `<service>` exists in `<ns>` with a ClusterIP, then the result will be a single A record with the specified name and the ClusterIP of that Service.
    - Answer Format:
      - `<service>.<ns>.svc.<zone>. <ttl> IN A <cluster-ip>`
    - Answer Example:
      - `kubernetes.default.svc.cluster.local. 4 IN A 10.3.0.1`
  - If a Service named `<service>` exists in `<ns>` without a ClusterIP (i.e., a headless service)  and there are no _ready_ endpoints for that Service, then the result will be empty (i.e., an empty Answer section).
  - If a Service named `<service>` exists in `<ns>` without a ClusterIP and there are _ready_ endpoints for the Service, the result will be an A record with the specified name and IP for each _ready_ endpoint.
    - Answer Format:
      - `<service>.<ns>.svc.<zone>. <ttl> IN A <endpoint-ip>`
    - Answer Example:
```
      headless.default.svc.cluster.local. 4 IN A 10.3.0.1
      headless.default.svc.cluster.local. 4 IN A 10.3.0.2
      headless.default.svc.cluster.local. 4 IN A 10.3.0.3
```
- **Question:** `<hostname>.<service>.<ns>.svc.<zone>. IN A`
  - If no Service named `<service>` exists in `<ns>`, the result will be `NXDOMAIN`.
  - If a Service named `<service>` exists in `<ns>` and a _ready_ endpoint exists for that Service with _hostname_ of `<hostname>` the result will be a single A record with the IP address of that endpoint. If no such endpoint exists, the result will be `NXDOMAIN`.
    - Answer Format:
      - `<hostname>.<service>.<ns>.svc.<zone>. <ttl> IN A <endpoint-ip>`
    - Answer Example:
      - `my-pet.my-service.default.svc.cluster.local. 4 IN A 10.3.0.100`
- **Question:** `<a>-<b>-<c>-<d>.<ns>.pod.<zone>. IN A`
  - If a Pod exists with IP address `<a>.<b>.<c>.<d>`, the result will be an A record with the specified name and the IP address `<a>.<b>.<c>.<d>`.
    - Answer Format:
      - `<a>-<b>-<c>-<d>.<ns>.pod.<zone>. <ttl> IN A <a>.<b>.<c>.<d>`
    - Answer Example:
      - `10-11-12-13.default.pod.cluster.local. 4 IN A 10.11.12.13`
  - If no such Pod exists, the result will be `NXDOMAIN`.
- **Question:** `_<port>._<proto>.<service>.<ns>.svc.<zone>. IN SRV`
  - If no Service named `<service>` exists in `<ns>`, the result will be `NXDOMAIN`.
  - If a Service named `<service>` exists in `<ns>` with a ClusterIP, and the service has a port named `<port>` using protocol `<proto>`, the result in the Answer section will be a single SRV record with the port number of the selected port and the name of the A record for the service (`<service>.<ns>.svc.<zone>.`). Unnamed ports will not receive an SRV record. The priority and weight returned are not prescribed by this specification. The Additional section may contain the named A record.
    - Answer Format:
       - `_<port>._<proto>.<service>.<ns>.svc.<zone>. <ttl> IN SRV <weight> <priority> <port-number> <service>.<ns>.svc.<zone>.`
    - Answer Example:
       - `_https._tcp.kubernetes.default.svc.cluster.local. 30 IN SRV 10 100 443 kubernetes.default.svc.cluster.local.`
  - If a Service named `<service>` exists in `<ns>` without a ClusterIP, and the Service has no ready endpoints, the result will be an empty Answer section.
  - If a Service named `<service>` exists in `<ns>` without a ClusterIP and with _ready_ endpoints, the result in the Answer section will be one SRV record for each _ready_ endpoint. Each record will contain the specific port number for the selected port and the name of an A record `<hostname>.<service>.<ns>.svc.<zone>` where `<hostname>` is the endpoint _hostname_. As described above, this name will resolve to an A record with the IP address of the specific _ready_ endpoint (which may be included in the Additional section). The priority and weight are not prescribed by this specification.
    - Answer Format:
      - `_<port>._<proto>.<service>.<ns>.svc.<zone>. <ttl> IN SRV <weight> <priority> <port-number> <hostname>.<service>.<ns>.svc.<zone>.`
    - Answer Example:
```
      _https._tcp.headless.default.svc.cluster.local. 4 IN SRV 10 100 443 my-pet-1.headless.default.svc.cluster.local.
      _https._tcp.headless.default.svc.cluster.local. 4 IN SRV 10 100 443 my-pet-2.headless.default.svc.cluster.local.
      _https._tcp.headless.default.svc.cluster.local. 4 IN SRV 10 100 443 438934893.headless.default.svc.cluster.local.
```
- **Question:** `<service>.<ns>.svc.<zone>. IN SRV`
  - If no Service named `<service>` exists in `<ns>`, the result will be `NXDOMAIN`.
  - If a Service named `<service>` exists in `<ns>` with a ClusterIP, and the service has **N** ports named `<port-1>, <port-2>, …, <port-n>` using protocol `<proto-1>, <proto-2>, …, <proto-n>` the result will be **N** SRV record with names `_<port-1>._<proto-1>.<service>.<ns>.svc.<zone>`, etc., the port number of the specific port and the name of the A record `<service>.<ns>.svc.<zone>`. The priority and weight returned are not prescribed by this specification.
    - Answer Format:
      - `_<port-x>._<proto-x>.<service>.<ns>.svc.<zone>. <ttl> IN SRV <weight> <priority> <port-number> <service>.<ns>.svc.<zone>.`
    - Answer Example:
```
      _https._tcp.my-service.default.svc.cluster.local.  4 IN SRV 10 100 443 my-service.default.svc.cluster.local.
      _http._tcp.my-service.default.svc.cluster.local.   4 IN SRV 10 100 80  my-service.default.svc.cluster.local.
      _syslog._udp.my-service.default.svc.cluster.local. 4 IN SRV 10 100 514 my-service.default.svc.cluster.local.
```
  - If a Service named `<service>` exists in `<ns>` without a ClusterIP and with _ready_ endpoints, the result in the Answer section will be one SRV record for each _ready_ endpoint and port combination. Thus, if there are **N** ports for the service and **M** _ready_ endpoints, there will be **N** :heavy_multiplication_x: **M** records in the Answer section. The name of each record will be `_<port>._<proto>.<service>.<ns>.svc.<zone>`. Here, `<port>` is the name of the port if it has one, and empty if it does not. `<proto>` is either “udp” or “tcp” depending on the port protocol. There will therefore be **M** records with the same name, one for each _ready_ endpoint. The records will contain the port number of the `<port>` and the name of an A record `<hostname>.<service>.<ns>.svc.<zone>` where `<hostname>` is the endpoint _hostname_. This name will resolve to an A record with the IP address of the specific _ready_ endpoint (which may be included in the Additional section). The priority and weight are not prescribed by this specification.
    - Answer Format:
      - `_<port-x>._<proto-x>.<service>.<ns>.svc.<zone>. <ttl> IN SRV <weight> <priority> <port-number> <service>.<ns>.svc.<zone>.`
    - Answer Example:
```
      _https._tcp.headless.default.svc.cluster.local.  4 IN SRV 10 100 443 my-pet-1.headless.default.svc.cluster.local.
      _http._tcp.headless.default.svc.cluster.local.   4 IN SRV 10 100 80  my-pet-1.headless.default.svc.cluster.local.
      _syslog._udp.headless.default.svc.cluster.local. 4 IN SRV 10 100 514 my-pet-1.headless.default.svc.cluster.local.
      _https._tcp.headless.default.svc.cluster.local.  4 IN SRV 10 100 443 my-pet-2.headless.default.svc.cluster.local.
      _http._tcp.headless.default.svc.cluster.local.   4 IN SRV 10 100 80  my-pet-2.headless.default.svc.cluster.local.
      _syslog._udp.headless.default.svc.cluster.local. 4 IN SRV 10 100 514 my-pet-2.headless.default.svc.cluster.local.
      _https._tcp.headless.default.svc.cluster.local.  4 IN SRV 10 100 443 438934893.headless.default.svc.cluster.local.
      _http._tcp.headless.default.svc.cluster.local.   4 IN SRV 10 100 80  438934893.headless.default.svc.cluster.local.
      _syslog._udp.headless.default.svc.cluster.local. 4 IN SRV 10 100 514 438934893.headless.default.svc.cluster.local.
```
- **Question:** `<reverse-ip-name>. IN PTR`
  - If a Service with ClusterIP equal to the IP of `<reverse-ip-name>` exists, a PTR record with `<service>.<ns>.svc.<zone>.` will be returned.
    - Answer Format:
      - `<reverse-ip-name>. <ttl> IN PTR <service>.<ns>.svc.<zone>.`
    - Answer Example:
      - `1.0.3.10.in-addr.arpa. 14 IN PTR kubernetes.default.svc.cluster.local.`
  - If a _ready_ endpoint exists with IP equal to the IP of `<reverse-ip-name>` exists, and the endpoint has a non-empty _hostname_ with value `<hostname>`, then a PTR record with `<hostname>.<service>.<ns>.svc.<zone>` will be returned.
    - Answer Format:
      - `<reverse-ip-name>. <ttl> IN PTR <hostname>.<service>.<ns>.svc.<zone>.`
    - Answer Example:
      - `100.0.3.10.in-addr.arpa. 14 IN PTR my-pet.my-service.default.svc.cluster.local.`
  - Otherwise, the result is `NXDOMAIN`.
- **Question:** `dns-version.<zone>. IN TXT`
  - Returns the semantic versioning of the DNS-schema in use in this cluster.
    - Answer Format:
      - `dns-version.<zone>. <ttl> IN TXT <schema-version>`
    - Answer Example:
      - `dns-version.cluster.local. 28800 IN TXT "1.0.0"`

## Schema Extensions

Specific implementations may choose to extend this schema, but the queries and responses described here are expected from any Kubernetes DNS solution.

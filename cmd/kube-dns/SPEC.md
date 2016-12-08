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

This document describes version 1.0 of the schema.

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
    - The value of the `HostName` field stored in the map indexed by the endpoint’s IP address in the `endpoints.beta.kubernetes.io/hostnames-map` annotation.
    - A unique, system-assigned identifier for the endpoint. The exact format and source of this identifier is not prescribed by this specification. However, it must be possible to use this to identify a specific endpoint in the context of a Service. This is used for SRV queries below in the event no explicit endpoint hostname is defined by the above methods.

- _ready_

 - An endpoint is considered _ready_ if its address is in the `addresses` field of the EndpointSubset object, or the corresponding service has the `service.alpha.kubernetes.io/tolerate-unready-endpoints` annotation set to `true`.

The supported queries and expected responses follow.

- **Type:** A, **Name:** `<service>.<ns>.svc.<zone>`

  - If no Service named `<service>` exists in `<ns>`, the result will be `NXDOMAIN`.

  - If a Service named `<service>` exists in `<ns>` with a ClusterIP, then the result will be a single A record with the specified name and the ClusterIP of that Service.

  - If a Service named `<service>` exists in `<ns>` without a ClusterIP (i.e., a headless service)  and there are no _ready_ endpoints for that Service, then the result will be empty (i.e., an empty Answer section).

  - If a Service named `<service>` exists in `<ns>` without a ClusterIP and there are _ready_ endpoints for the Service, the result will be an A record with the specified name and IP for each _ready_ endpoint.

- **Type:** A, **Name:** `<hostname>.<service>.<ns>.svc.<zone>`

  - If no Service named `<service>` exists in `<ns>`, the result will be `NXDOMAIN`.

  - If a Service named `<service>` exists in `<ns>` and a _ready_ endpoint exists for that Service with _hostname_ of `<hostname>` the result will be a single A record with the IP address of that endpoint. If no such endpoint exists, the result will be `NXDOMAIN`.

- **Type:** A, **Name:** `<a>-<b>-<c>-<d>.<ns>.pod.<zone>`

  - If a Pod exists with IP address `<a>.<b>.<c>.<d>`, the result will be an A record with the specified name and the IP address `<a>.<b>.<c>.<d>`.

  - If no such Pod exists, the result will be `NXDOMAIN`.

- **Type:** SRV, **Name:** `_<port>._<proto>.<service>.<ns>.svc.<zone>`

  - If no Service named `<service>` exists in `<ns>`, the result will be `NXDOMAIN`.

  - If a Service named `<service>` exists in `<ns>` with a ClusterIP, and the service has a port named `<port>` using protocol `<proto>`, the result in the Answer section will be a single SRV record with the port number of the selected port and the name of the A record for the service (`<service>.<ns>.svc.<zone>`). If the service has exactly one port and it is unnamed, then `<port>` may be empty. The priority and weight returned are not prescribed by this specification. The Additional section may contain the named A record.

  - If a Service named `<service>` exists in `<ns>` without a ClusterIP, and the Service has no ready endpoints, the result will an empty Answer section.

  - If a Service named `<service>` exists in `<ns>` without a ClusterIP and with _ready_ endpoints, the result in the Answer section will be one SRV record for each _ready_ endpoint. Each record will contain the specific port number for the selected port and the name of an A record `<hostname>.<service>.<ns>.svc.<zone>` where `<hostname>` is the endpoint _hostname_. As described above, this name will resolve to an A record with the IP address of the specific _ready_ endpoint (which may be included in the Additional section). The priority and weight are not prescribed by this specification.

- **Type:** SRV, **Name:** `<service>.<ns>.svc.<zone>`

  - If no Service named `<service>` exists in `<ns>`, the result will be `NXDOMAIN`.

  - If a Service named `<service>` exists in `<ns>` with a ClusterIP, and the service has **N** ports named `<port-1>, <port-2>, …, <port-n>` using protocol `<proto-1>, <proto-2>, …, <proto-n>` the result will be **N** SRV record with names `_<port-1>._<proto-1>.<service>.<ns>.svc.<zone>`, etc., the port number of the specific port and the name of the A record `<service>.<ns>.svc.<zone>`. The priority and weight returned are not prescribed by this specification.


  - If a Service named `<service>` exists in `<ns>` without a ClusterIP and with _ready_ endpoints, the result in the Answer section will be one SRV record for each _ready_ endpoint and port combination. Thus, if there are **N** ports for the service and **M** _ready_ endpoints, there will be **N** :heavy_multiplication_x: **M** records in the Answer section. The name of each record will be `_<port>._<proto>.<service>.<ns>.svc.<zone>`. Here, `<port>` is the name of the port if it has one, and empty if it does not. `<proto>` is either “udp” or “tcp” depending on the port protocol. There will therefore be **M** records with the same name, one for each _ready_ endpoint. The records will contain the port number of the `<port>` and the name of an A record `<hostname>.<service>.<ns>.svc.<zone>` where `<hostname>` is the endpoint _hostname_. This name will resolve to an A record with the IP address of the specific _ready_ endpoint (which may be included in the Additional section). The priority and weight are not prescribed by this specification.

- **Type:** PTR, **Name:** `<reverse-ip-name>`

  - If a Service with ClusterIP equal to the IP of `<reverse-ip-name>` exists, a PTR record with `<service>.<ns>.svc.<zone>` will be returned.

  - If a _ready_ endpoint exists with IP equal to the IP of `<reverse-ip-name>` exists, and the endpoint has a non-empty _hostname_ with value `<hostname>`, then a PTR record with `<hostname>.<service>.<ns>.svc.<zone>` will be returned.

  - Otherwise, the result is `NXDOMAIN`.

- **Type:** TXT, **Name:** `dns-version.<zone>`

  - Returns the semantic versioning of the DNS-schema in use in this cluster. For example, “1.0”.

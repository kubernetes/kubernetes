# Package: apis

## Purpose
The `apis` package defines well-known labels and constants used by kube-proxy.

## Constants

- **LabelServiceProxyName**: `service.kubernetes.io/service-proxy-name`
  - Label indicating that an alternative service proxy will implement a Service.
  - Services with this label are ignored by the default kube-proxy.
  - Allows third-party proxy implementations to handle specific services.

## Design Notes

- Enables multi-proxy deployments where different proxies handle different services.
- Used for custom load balancers or service mesh implementations.
- Standard kube-proxy skips services with this label set.

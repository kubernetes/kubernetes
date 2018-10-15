# kubernetes-csi-migration-library

Library of functions to be consumed by various Kubernetes and CSI components in
order to support In-tree plugin to CSI Migration.

To use, import this library into the component that requires source translation
and use the `TranslateToCSI` and `TranslateToInTree` functions.

This library has a couple requirements:
1. The driver must have a stable (well-known) driver name
2. The translation library must not assume its running location
    1. This mean's no access to Kubernetes API Server
    2. Assume no network connectivity (no cloud APIs)
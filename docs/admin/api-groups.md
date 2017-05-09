# API groups

Kubernetes provides multiple API groups. API groups each have potentially
multiple versions. Except for the v1 API, group/version pairs are formated like
"group/version", for example "experimental/v1alpha1". The v1 API version is in
the group "", and is written just "v1".

## Turning groups on or off

These can be enabled or disabled by changing `kube-apiserver`'s
`--runtime-config` flag. For example, `v1=true,experimental/v1alpha1=false`
enables the normal API and disables the experimental API.

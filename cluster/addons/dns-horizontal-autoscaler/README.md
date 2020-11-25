# DNS Horizontal Autoscaler

DNS Horizontal Autoscaler enables horizontal autoscaling feature for DNS service
in Kubernetes clusters. This autoscaler runs as a Deployment. It collects cluster
status from the APIServer, horizontally scales the number of DNS backends based
on demand. Autoscaling parameters could be tuned by modifying the `kube-dns-autoscaler`
ConfigMap in `kube-system` namespace.

Learn more about:
- Usage: http://kubernetes.io/docs/tasks/administer-cluster/dns-horizontal-autoscaling/
- Implementation: https://github.com/kubernetes-incubator/cluster-proportional-autoscaler/

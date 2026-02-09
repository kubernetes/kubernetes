# Calico Policy Controller

Calico is an implementation of the Kubernetes network policy API.  The provided manifests install:

- A DaemonSet which runs Calico on each node in the cluster.
- A Deployment which installs the Calico Typha agent.
- A Service for the Calico Typha agent.
- Horizontal and vertical autoscalers for Calico.

### Learn More

Learn more about Calico at https://docs.projectcalico.org


# Clustering in Kubernetes

### Overview
There are multiple different ways to achieve clustering with different security and usability profiles.  This document attempts to lay out the user experiences for clustering that Kubernetes aims to address

### Centralized clustering
In centralized clustering, a single entity (generally the master) is responsible for determining the set of machines which are members of the cluster.  Calls to create and remove worker nodes in the cluster are restricted to this single authority, and any other requests to add or remove worker nodes are rejected.  Centralized clustering is the least flexible approach to clustering, since all additions or removal of worker nodes must go through the central authority, it is however, the most secure form of clustering, since there is only a single entity that is capable adding worker nodes to the cluster.

### Discovery clustering
In discovery clustering there is a known cluster discovery rally point which is pre-defined for the cluster.  This rally point can be used to obtain credentials that enable a worker node to join the cluster.  Thus the credentials and credential distribution are still centralized, but the requests to add or removed nodes are performed by the nodes themselves.  Discovery clustering is somewhat more flexible than centralized clustering, because it allows nodes to use a centralized service to obtain appropriate credentials, but because the authorization is obtained from a centralized source, it is also easier to prevent unknown nodes from joining the cluster.

### Dynamic clustering
In dynamic clustering, anyone is free to propose membership into the cluster, and the decision about whether or not to accept the membership is made dynamically by the master at the time of the request.  This is the most flexible approach, but also has serious security concerns, as the master must be very careful to not allow malicious nodes to join the cluster and steal work (or the data contained in the work)
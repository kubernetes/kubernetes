## Refactor Cloud Provider out of Kubernetes Core

As kubernetes has evolved tremendously, it has become difficult for different cloudproviders (currently 7) to make changes and iterate quickly. Moreover, the cloudproviders are constrained by the kubernetes build/release lifecycle. It has also become harder to add new cloud providers. This proposal aims to move towards a kubernetes code base where cloud providers specific code will move out of the core repository, and will be maintained by the cloud providers themselves.

### 1. Current use of Cloud Provider

The following components have cloudprovider dependencies

    1. kube-controller-manager
    2. kubelet
    3. kube-apiserver

#### Cloud Provider in Kube-Controller-Manager

The kube-controller-manager has many controller loops

 - nodeController
 - volumeController
 - routeController
 - serviceController
 - replicationController
 - endpointController
 - resourceQuotaController
 - namespaceController
 - deploymentController
 - etc..

Among these controller loops, the following are cloud provider dependent.

 - nodeController
 - volumeController
 - routeController
 - serviceController

The nodeController uses the cloudprovider to check if a node has been deleted from the cloud. If cloud provider reports a node as deleted, then this controller immediately deletes the node from kubernetes. This check removes the need to wait for a specific amount of time to conclude that an inactive node is actually dead.

The volumeController uses the cloudprovider to create, delete, attach and detach volumes to nodes. For eg, the logic for provisioning, attaching, and detaching a EBS volume resides in the AWS cloudprovider. The volumeController uses this code to perform its operations.

The routeController configures routes for hosts in Google Cloud.

The serviceController maintains a list of currently active nodes, and is responsible for creating and deleting LoadBalancers in the underlying cloud.

#### Cloud Provider in Kubelet

Moving on to the kubelet, the following cloud provider dependencies exist in kubelet.

 - Find the cloud nodename of the host that kubelet is running on for the following reasons :
      1. To obtain the config map for the kubelet, if one already exists
      2. To uniquely identify current node using nodeInformer
      3. To instantiate a reference to the current node object
 - Find the InstanceID, ProviderID, ExternalID, Zone Info of the node object while initailizing it
 - Periodically poll the cloud provider to figure out if the node has any new IP addresses associated with it
 - In case of Google cloud, it allows the provider to configure routes for the node
 - It allows the cloud provider to post process DNS settings

#### Cloud Provider in Kube-apiserver

Finally, in the kube-apiserver, the cloud provider is used for trasferring SSH keys to all of the nodes, and within an admission controller for setting labels on persistent volumes.

### 2. Strategy for refactoring Kube-Controller-Manager

In order to create a 100% cloud independent controller manager, we will split the controller-manager into multiple binaries.

1. Cloud dependent controller-manager binaries
2. Cloud independent controller-manager binaries

The cloud dependent binaries will run those loops that rely on cloudprovider. The rest of the controllers will be run in the cloud independent controller manager. The decision to run entire controller loops, rather than only the very minute parts that rely on cloud provider makes implementation simple. If drilled down further into the controller loops, it would be significantly harder to disentangle, due to tight coupling of the cloud provider specific code with the rest of the data structures/functions within a controller. There would be a lot of duplicated code, which needs to be kept in sync, across the "dependent" loop implementations. For example, StatefulSet would not have gotten to beta in 1.5 if we had to change 7 node controller loops in various cloud provider binaries.

Note that the controller loop implementation will continue to reside in the core repository. It takes in cloudprovider.Interface as an input in its constructor. Vendor maintained cloud-controller-manager binary will link these controllers in. It will also initialize the cloudprovider object and pass it into the controller loops.

There are four controllers that rely on cloud provider specific code. These are node controller, service controller, route controller and attach detach controller. Copies of each of these controllers have been bundled them together into one binary. This binary registers itself as a controller, and runs the cloud specific controller loops with the user-agent named "external-controller-manager".

RouteController and serviceController are entirely cloud specific. Therefore, it is really simple to run these two controller loops separately.

NodeController does a lot more than just talk to the cloud. It does the following operations -

1. CIDR management
2. Monitor Node Status
3. Node Pod Eviction

While Monitoring Node status, if the status reported by kubelet is either 'ConditionUnknown' or 'ConditionFalse', then the controller checks if the node has been deleted from the cloud provider. If it has already been deleted from the cloud provider, then it deletes the nodeobject without waiting for the `monitorGracePeriod` amount of time. This operation will be moved into the cloud specific controller manager.

Finally, The attachDetachController is tricky, and it is not simple to disentangle it from the controller-manager easily, therefore, this will be addressed with Flex Volumes (Discussed under a separate section below)

### 3. Strategy for refactoring Kubelet

The majority of the calls by the kubelet to the cloud is done during the initialization of the Node Object. The other uses are for configuring Routes (in case of GCE), scrubbing DNS, and periodically polling for IP addresses.

All of the above steps, except the Node initialization step can be moved into a controller. Specifically, IP address polling, and configuration of Routes can be moved into the cloud dependent controller manager.

Scrubbing DNS, after discussing with @thockin, was found to be redundant. So, it can be disregarded. It is being removed.

Finally, Node initialization needs to be addressed. This is the trickiest part. Pods will be scheduled even on uninitialized nodes. This can lead to scheduling pods on incompatible zones, and other wierd errors. Therefore, an approach is needed where kubelet can create a Node, but mark it as "NotReady". Then, some asynchronous process can update it and mark it as ready. This is now possible because of the concept of ExternalAdmissionControllers introduced by @justinsb in these PRs - (https://github.com/kubernetes/kubernetes/pull/36210, https://github.com/kubernetes/kubernetes/pull/36209)

Admission controllers are meant to perform some aspects of initialization of objects. In this case, an ExternalAdmissionController will be configured to first mark the Node object as unusable by setting the NodeStatus as {"CloudProviderInitialized": "False"} (This exact keyword has not been decied yet), and then an external controller (residing in the cloud controller manager) will update it by removing the NodeStatus. In order for this to work, Nodes should be unusable until these Statuses are removed. This is possible because of the above mentioned PRs by @justinsb.

### 4. Strategy for refactoring Kube-ApiServer

Kube-apiserver uses the cloud provider for two purposes

1. Distribute SSH Keys - This can be moved to a controller
2. Admission Controller for PV - This can be refactored using the External Admission Controller approach used in Kubelet

### 5. Strategy for refactoring Volumes

Volumes need cloud providers, but they only need SPECIFIC cloud providers.  The real problem is that cloud providers can take params from cloud config. The majority of the logic for volume management, currently resides in the controller-manager.

There is an undergoing effort to move all of the volume logic from the controller-manager into plugins called Flex Volumes. In the Flex volumes world, all of the vendor specific code will be packaged in a separate binary as a plungin. After discussing with @thockin, this was decidedly the best approach to remove all cloud provider dependency for volumes out of kubernetes core.

### 6. Roadmap

As the first step, an opt-in flag will be provided to run the external cloud controller manager. This is being aimed to be completed by 1.6 release.

Then in 1.7 or 1.8, Flex Volumes will replace traditional volume management. This will make it possible to refactor volume oriented cloudprovider code from kubernetes. In this period, non-core cloud vendors can start running kubernetes natively in their cloud (without volumes)

Meanwhile, various vendors will be notified to update their cloudprovider implementation to the new style. Finally, external cloud controller manager will be the default way to run kubernetes.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/cloud-provider-refactoring.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

# Kubernetes Documentation

* [Primary concepts](#primary-concepts)
* [Further reading](#further-reading)


Getting started guides are in [getting-started-guides](getting-started-guides).

There are example files and walkthroughs in the [examples](../examples) folder.

If you're developing Kubernetes, docs are in the [devel](devel) folder.

Design docs are in [design](design).

API objects are explained at [http://kubernetes.io/third_party/swagger-ui/](http://kubernetes.io/third_party/swagger-ui/).

Frequently asked questions are answered on this project's [wiki](https://github.com/GoogleCloudPlatform/kubernetes/wiki).

## Primary concepts

* **Overview** ([overview.md](overview.md)): A brief overview
  of Kubernetes concepts. 

* **Nodes** ([node.md](node.md)): A node is a worker machine in Kubernetes.

* **Pods** ([pods.md](pods.md)): A pod is a tightly-coupled group of containers
  with shared volumes.

* **The Life of a Pod** ([pod-states.md](pod-states.md)):
  Covers the intersection of pod states, the PodStatus type, the life-cycle
  of a pod, events, restart policies, and replication controllers.

* **Replication Controllers** ([replication-controller.md](replication-controller.md)):
  A replication controller ensures that a specified number of pod "replicas" are 
  running at any one time.

* **Services** ([services.md](services.md)): A Kubernetes service is an abstraction 
  which defines a logical set of pods and a policy by which to access them.

* **Volumes** ([volumes.md](volumes.md)): A Volume is a directory, possibly with some 
  data in it, which is accessible to a Container.

* **Labels** ([labels.md](labels.md)): Labels are key/value pairs that are 
  attached to objects, such as pods. Labels can be used to organize and to 
  select subsets of objects. 

* **Accessing the API** ([accessing_the_api.md](accessing_the_api.md)):
  Ports, IPs, proxies, and firewall rules.

* **Kubernetes Web Interface** ([ui.md](ui.md)): Accessing the Kubernetes
  web user interface.

* **Kubectl Command Line Interface** ([kubectl.md](kubectl.md)):
  The `kubectl` command line reference.

* **Roadmap** ([roadmap.md](roadmap.md)): The set of supported use cases, features,
  docs, and patterns that are required before Kubernetes 1.0.

* **Glossary** ([glossary.md](glossary.md)): Terms and concepts.


## Further reading


* **Annotations** ([annotations.md](annotations.md)): Attaching
  arbitrary non-identifying metadata.

* **API Conventions** ([api-conventions.md](api-conventions.md)):
  Defining the verbs and resources used in the Kubernetes API.

* **Authentication Plugins** ([authentication.md](authentication.md)):
  The current and planned states of authentication tokens.

* **Authorization Plugins** ([authorization.md](authorization.md)):
  Authorization applies to all HTTP requests on the main apiserver port.
  This doc explains the available authorization implementations.

* **API Client Libraries** ([client-libraries.md](client-libraries.md)):
  A list of existing client libraries, both supported and user-contributed.

* **Kubernetes Container Environment** ([container-environment.md](container-environment.md)):
  Describes the environment for Kubelet managed containers on a Kubernetes
  node.

* **DNS Integration with SkyDNS** ([dns.md](dns.md)):
  Resolving a DNS name directly to a Kubernetes service.

* **Identifiers** ([identifiers.md](identifiers.md)): Names and UIDs
  explained.

* **Images** ([images.md](images.md)): Information about container images
  and private registries.

* **Logging** ([logging.md](logging.md)): Pointers to logging info.

* **Namespaces** ([namespaces.md](namespaces.md)): Namespaces help different
  projects, teams, or customers to share a kubernetes cluster.

* **Networking** ([networking.md](networking.md)): Pod networking overview.

* **OpenVSwitch GRE/VxLAN networking** ([ovs-networking.md](ovs-networking.md)):
  Using OpenVSwitch to set up networking between pods across
  Kubernetes nodes.

* **The Kubernetes Resource Model** ([resources.md](resources.md)):
  Provides resource information such as size, type, and quantity to assist in
  assigning Kubernetes resources appropriately.

* **Using Salt to configure Kubernetes** ([salt.md](salt.md)): The Kubernetes
  cluster can be configured using Salt.


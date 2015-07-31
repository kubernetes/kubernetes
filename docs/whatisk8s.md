<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/whatisk8s.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# What is Kubernetes?

Kubernetes is an open-source platform for automating deployment, scaling, and operations of application containers across clusters of hosts.

With Kubernetes, you are able to quickly and efficiently respond to customer demand:

 - Scale your applications on the fly.
 - Seamlessly roll out new features.
 - Optimize use of your hardware by using only the resources you need.

#### Kubernetes is:

* **lean**: lightweight, simple, accessible
* **portable**: public, private, hybrid, multi-cloud
* **extensible**: modular, pluggable, hookable, composable
* **self-healing**: auto-placement, auto-restart, auto-replication

The Kubernetes project was started by Google in 2014. Kubernetes builds upon a [decade and a half of experience that Google has with running production workloads at scale](https://research.google.com/pubs/pub43438.html), combined with best-of-breed ideas and practices from the community.

##### Ready to [Get Started](getting-started-guides/README.md)?

<hr>

Looking for reasons why you should be using [containers](http://aucouranton.com/2014/06/13/linux-containers-parallels-lxc-openvz-docker-and-more/)?

Here are some key points:

* **Application-centric management**:
    Raises the level of abstraction from running an OS on virtual hardware to running an application on an OS using logical resources. This provides the simplicity of PaaS with the flexibility of IaaS and enables you to run much more than just [12-factor apps](http://12factor.net/).
* **Dev and Ops separation of concerns**:
    Provides separatation of build and deployment; therefore, decoupling applications from infrastructure.
* **Agile application creation and deployment**:
    Increased ease and efficiency of container image creation compared to VM image use.
* **Continuous development, integration, and deployment**:
    Provides for reliable and frequent container image build and deployment with quick and easy rollbacks (due to image immutability).
* **Loosely coupled, distributed, elastic, liberated [micro-services](http://martinfowler.com/articles/microservices.html)**:
	Applications are broken into smaller, independent pieces and can be deployed and managed dynamically -- not a fat monolithic stack running on one big single-purpose machine.
* **Environmental consistency across development, testing, and production**:
    Runs the same on a laptop as it does in the cloud.
* **Cloud and OS distribution portability**:
    Runs on Ubuntu, RHEL, on-prem, or Google Container Engine, which makes sense for all environments: build, test, and production.
* **Resource isolation**:
    Predictable application performance.
* **Resource utilization**:
    High efficiency and density.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/whatisk8s.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

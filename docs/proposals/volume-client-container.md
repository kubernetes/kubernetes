# Containerized Volume Clients for Kubernetes
This document presents user stories, design and examples for containerized volume clients.

# Motivation
Many storage systems require specific software to connect.  The current volume architecture requires volume client software installed on each host for the kubelet to access volumes.  Its an common mistake to forget to install the packages on one or more hosts.  In addition, its often difficult to install the packages potentially impossible on some OS.

1. No client installed on host
2. No client configuration on host.


# Roles and Definitions
* **Platform** Host environment running kubernetes (Ex: Fedora-atomic, CentOS)
* **Platform Maintainer** - Maintains host platform.
* **Framework** - Kubernetes or applications packaging Kubernetes
* **Framework Maintainer** - Builds, packages, bundles or distributes framework.
* **Storage Framework Maintainer** - Owns or distributes a specific storage solution.
* **Storage Administrator** - Creates, owns and adminsters a deployed storage system independent of kubernetes.
* **Kubernetes Administrator** - Maintains Kubernetes cluster with elevated permissions compared to User.
* **User** - End user of Kubernetes.  Submits applications to running framework.
* **Tire Kicker** - Single/simple deployer of Kubernetes.  Runs example, non production use cases with the combined responsibilities of Storage, Kube and Platform administrator and end user.
* **VCS** - Volume (Client/Control) Software. Any packages or utilities required to mount/read/write/admin a storage volume. Provided by host platform.
* **Storage Control Plane** - Storage Administrator operation interface for create/delete/modify type functions of the storage subsystem external from Kubernetes. Example: Browser interface to SAN fabric configuration.


# User Stories

* ** Story 0**: As a Framework maintainer, I want to provide a no-setup storage experience for non-production environments.

* ** Story 0**: As a Framework maintainer, I want to allow custom configuration for production environments.

* **Story  1**: As a Kubernetes Administrator I do not want to install volume drivers.

* **Story  2**: As a Framework Maintainer I want to reduce kube configuration required to use kubernetes storage.

* **Story  2**: As a Framework Maintainer I want to reduce/eliminate platform configuration required to use storage.

* **Story  3**: As a Framework Maintainer I want to provide VCS support for volume types that works out-of-the-box.

* **Story  4**: As a Framework Maintainer I want a workable zero-configuration out-of-the-box experience.

* **Story  4**: As a Framework Maintainer I want applications to be portable across Platforms.

* **Story  4**: As a Framework Maintainer I want supported volumes to work regardless of Platform.

* **Story  5**: As a Storage Framework Maintainer I want to provide enhanced or custom supported VCS.

* **Story  6**: As a Storage Framework Maintainer I want to provide out-of-tree VCS.

* **Story  6**: As a Storage Administrator I want a deliberate upgrade path for VCS.

* **Story  6**: As a Storage Framework Maintainer my storage system requires the VCS to continue running while the storage is being used. (TSR/Daemon requried for client)

* **Story  6**: As a Platform Matinaer I want the VCS containerized

* **Story  6**: As a Platform Maintainer I do not want the VCS containerized or managed by the Framework. The platform will manage the VCS.

* **Story  6**: As a tire-kicker I want to run network storage examples with no additional configuration.

* **Story  6**: As a tire-kicker I want to run network storage examples with no manual VCS configuration



# Design
Some storage volumes are supported by default in the host kernel but others require specific software packages (VCS). The VCS is typically dependent on the host platform, which presents configuration and administration difficulties for the Framework Maintainer.

By containerizing the VCS they become portable across multiple platforms. A containerized VCS can mount the volume then export the mount to other containers eliminating the need for VCS installed on the host.

# Mount Namespace
The platform maintains a root mount namespace and each new user or container instance creates a slave namespace.  Mounts in the root namespace are visible to slaves, but slave namespace mounts are not visible to the root or other slave mount namespaces by default.

With Docker 1.10 its possible to share a slave mount namespace with the root namespace.  By sharing the namespace, one container can mount a filesystem thats visible by another container.

# Configuration
As of Kubernetes 1.2, the volume plugins expect that the VCS is installed on the host.  *Each plugin will have to opt-in to container usage* this may leave a undeseriable "out of the box" experience until the default is changed to opt-out. This design prsents a pattern which will be added to plugins independently.

For greater flexibility each plugin will accept a CLI option to specify the container or host to mount.  If container not specified host is assumed.  If the default chnages to always user a container then the CLI arg will take 'host' as an additional option  Example:

--nfs_mount_container="k8/nfs_container:latest"

or

--nfs_mount_container="host"

# Container State #
The volume plugin will manage the containers state if necesarry.  If a container must remain running while the mount is available, the container should ensure as such.  Once the volume is cleaned the container may be released.

# Container Design
In many cases its as simple as creating a container with the entire VCS and executing the same mount command that would have been executed on host.  This allows a single mount command to work both on a host VCS or containerized VCS.

With other scenarios it may be desirable to override the mount command, or continue running.  Each in-tree plugin should handle this on a case-by-case basis.

## Creation of Provisioner
The Storage Administrator is responsible for creating a provisioning container.  The details and contents of the provisioning container aren't important only that it creates a new Persistent Volume before exit.  This requires *kubectl* on the provisioner and a service account token passed into the container.

Storage Administrator provides Kubernetes Adminstrator with a provisioning container. Kubernetes Adminstrator ensures the container is available on appropriate nodes.  Properties passed into the provisioner could be on a secrets volume or as environment variables which can be parsed by shell scripts or binaries within the provisioning container.  

To support Use Case #10, the flex provisioner may accept a node selector to target specific nodes for running the provisioner.  If no node-selector is specified then any available node may be used.  Containers will run as pods.

## Provisioner Config
The kubernetes admin is responsible for configuring kubernetes to use the provisioner.  Configuration for custom provisioners is similar to configuration of storage-claseses using a ConfigMap.  

A custom provisioner ConfigMap will require an additional field to specify a container which will perform the provisioning work.  Extra paramaters can be arbitrarily set by the kubernetes admin during configurion which will be passed to the container as well as properties provided by the API.  Such API properties include user, group, size, PV-name, storage-class and service account details.  Service account is used to create the PV by the provisioning container.

**Example Config**: 
Here is an example markup file that the Kubernetes Administrator would use to configure a provisioner:

```yaml
kind: ConfigMap
metadata:
  name: custom-isci-provisioner
  namespace: admin
  labels:
    storage-class: silver
    type: provisioner-config
data:
  plugin-name: kubernetes.io/flex-provisioner
  image: custom/iscsi-provision
  service-account-name: foo
  service-token:23092390532lksf90kwelklfw
  use-nfs-server:10.10.2.4
  zone:west
``` 
**Example Properties**: Based on the above example config these are the properties passed from the Flex Provisioner to the provisioning POD when user tom requests a new storage-class silver volume:

Property | Value 
--- | ---
pv-name | something-pvy
username | tom
storage-class| silver
zone | west
use-nfs-server|10.10.2.4
service-account|foo
service-token |23092390532lksf90kwelklfw
kube-api | 10.10.2.1

## Provisioner Completion and Exit
After the provisioner terminates it must return error or success.  A success return indicates that the new persistent volume is created.  Its up to the provisioning framework to handle a provisioning success or failure.

## Sequence of Events
     
***Call Flow***: User submits PVC --> Apiserver--> Dynamic Provisioning Framework--> Flex Provisioner--> Provisioning Pod--> Storage Control Plane

***Return Flow***: Storage Control Plane--> Provisioning Pod--(PV created via kubectl)--> Api-server

Once the PV is created the PVC will attach by existing mechanism completing the flow.

# Function Examples

**Storage Administrator** - Creating a Custom Provisioner:

Dockerfile:

``` Dockerfile
FROM centos
RUN yum update ; yum install iscsi-initiator-utils
RUN curl www.examples.com/me/iscsi-provision.sh >> /usr/bin/iscsi-provision.sh
ENTRYPOINT ["/usr/bin/iscsi-provision.sh"]
```
docker build . -t custom/iscsi-provision:latest
 
**Kubernetes Administrator** - Configuring a custom provisioner for silver storage-class:
```yaml
kind: ConfigMap
metadata:
  name: custom-isci-provisioner
  namespace: admin
  labels:
    storage-class: silver
    type: provisioner-config
data:
  plugin-name: kubernetes.io/flex-provisioner
  image: custom/iscsi-provision
  service-account-name: foo
  service-token:23092390532lksf90kwelklfw
  use-nfs-server:10.10.2.4
  zone:west
```

**Kubernetes User** - Same workflow as standard provisioner.

# Technical Motivation of DPF design change
* The proposed design is asynchronous
* No orphaned PV objects
* Delegation of recovery to provisioner.  For out-of-tree provisioners this allows greater flexability.
* No need to re-write volume template in the Provision() call which requires specific volume object Types. The current limitation restricts provisioners to volumes of their plugin type.  So no out of tree NFS provisioning.


# Design Alternatives
**Alternative 1**: Use a binary/shell script on the master instead of a POD.  This violates Use Case  9 and possibly Use Case 5.  It also complicates the passing of properties from kubernets API to provisioners.




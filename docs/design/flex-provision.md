# Flex Storage Provisioner for Kubernetes
This document presents use cases, design and examples for a flexible storage provisioner in Kubernetes.  

# Motivation
Storage provisioning is often a filesystem and site specific behavior.  Its impossible to anticpate all storage provisioning scenarios and maintain numerous provisioners in-source.  Storage vendors often prefer working on and owning provisioning layers outside of the Kubernetes source while maintaining compatability.  This design will present additional benefits such as:

1. Field configuration and adjustment of provisioners.
2. Greater flexibility in provisioning function.
3. Pluggable support for filesystem provisioning.
4. Independent upgrade path of Kubernetes and flex storage provisioners.
5. Significantly lowered barrier-to-entry for additional filesystem types wishing to provide provisioning.

# Roles and Definitions
* **Storage Administrator** - Creates, owns and adminsters storage systems independent of kubernetes.
* **Kubernetes Administrator** - Maintains Kubernetes cluster with elevated permissions compared to User.
* **User** - End user of Kubernetes.  Submits applications to running framework.
* **PV** - Persistent Volume
* **PVC** - Persistent Volume Claim
* **DPF** - Dynamic Provisioning Framework
* **Storage Control Plane** - Storage Administrator operation interface for create/delete/modify type functions of the storage subsystem external from Kubernetes. Example: Browser interface to SAN fabric configuration.



# Use Cases
* **Case  1**: As a Kubernetes Administrator I wish to provision volumes that aren't supported by kubernetes core. 

* **Case  2**: As a Kubernetes Administrator I want to perform extra steps while the storage is provisioned such as copying data, notifying an admin or creating directories before attaching it to a volume.

* **Case  2**: As a Storage Administrator I want to provide a provisioner for my storage solution which may include proprietary details that I dont want public or requires code that exists under a non-compatible license.

* **Case  3**: As a Storage Administrator I want to provide a storage provisioner that is public but I dont want to be part of the kubernetes release cycle.

* **Case  4**: As a Storage Administrator I want to provide maintenence and support of my provisioner in my own bug tracking and source control.

* **Case  5**: As a Kubernetes Administrator I'd like to keep the operating system storage packages off the host nodes.

* **Case  6**: As a Kubernetes and Storage Administrator I wish to enable provisioning with existing tools or constructs i'm already familiar with such as bash.

* **Case  7**: As a Kubernetes Administrator I want to use the same provisioner for multiple clases of storage without re-writing the provisioner.

* **Case  8**: ~~As a Kubernetes Administrator I want the naming pattern of provisioned persistent volumes to follow a custom convention.~~ ***Note***: pv name established by Dynamic Provisioning framework no option to change.

* **Case  9**: As a Storage Administrator I want to deliver provisioners to the Kubernetes Administrator and provide updates/changes in as automated fashion as possible.

* **Case 10**: As a Kubernetes Administrator I want to secure the node performing provisioning from the rest of the network.


# Design
The proposed design alters the Dynamic Provisioning Frameworks API to allow asynchronous volume provisioning and faciliate generic volume creation by provisioners.  

The current flow requires a provisioner plugin to provide a volume template which the DPF uses to create and pre-bind the volume to claim before explicitly creating the phsyical storage asset.  The DPF waits for the provisioner to complete and return the real PV before updating the previously created&bound PV with the new PV properties.

In the proposed new flow the DPF would pass a unique ID to the provisioner plugin. This unique ID is either an existing (CLAIMREF?) PVC field or add a new PVC field for UUID.   The provisioner plugin then creates the physical storage asset and a new PV with an ANNOTATION of CLAIM_UUID as passed in by the DPF.  The claim binding controller watch will trigger and bind the original claim to the newly created PV based on the CLAIMREF annotation.  A new PROVISION_TS field is required on the PVC object to indicate the last provisioning attempt for cluster restart/recovery.

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




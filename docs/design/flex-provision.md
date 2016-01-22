# Flex Storage Provisioner in Kubernetes
This document presents use cases, design and examples for a flexible dynamic provisioner in Kubernetes.  

# Motivation
Storage provisioning is often a filesystem and site specific behavior.  Its difficult or impossible to anticpate all storage provisioning scenarios and maintain all possible provisioners in-source.  Storage vendors often prefer working on and owning provisioning layers outside of the Kubernetes source while maintaining compatability.  This design will present additional benefits such as:

1. In field configuration and adjustment of provisioners.
2. Greater flexibility in provisioning function.
3. Pluggable support for private filesystem provisioning.
4. Independent upgrade path of Kubernetes and flex Storage Provisioners
5. Significantly lowered barrier-to-entry for new filesystem types wishing to provide provisioning.

# Roles and Definitions
* **Storage Administrator** - Creates, owns and adminsters storage systems independent of kubernetes.
* **Kubernetes Administrator** - Sets up and maintains Kubernetes.  Has elevated permissions compared to standard users.
* **User** - End user of Kubernetes.  Submits applications to the running framework.
* **PV** - Persistent Volume
* **PVC** - Persistent Volume Claim
* **Storage Control Plane** - Storage Administrator operation interface for create/delete/modify type functions of the storage subsystem 0external from Kubernetes. Example: Browser interface to SAN fabric configuration.



# Use Cases
* **Case  1**: As a kubernetes administrator I wish to dynamically provision volumes that aren't supported by kubernetes core. 

* **Case  2**: As a kubernetes administrator I want to perform extra steps after my volume is provisioned such as copying data, notifying an admin or creating directories before attaching it to a volume.

* **Case  2**: As a storage administrator I want to provide a provisioner for my storage solution which may include proprietary details that I dont want public or requires code that exists under a non-compatible license.

* **Case  3**: As a storage administrator I want to provide a storage provisioner that is public but I dont want to be part of the kubernetes release cycle.

* **Case  4**: As a storage administrator I want to provide maintenence and support of my provisioner in my own bug tracking and source control.

* **Case  5**: As a kubernetes administrator I'd like to keep the operating system storage packages off my host nodes.

* **Case  6**: As a kubernetes and storage administrator I dont know golang and wish to enable provisioning with constructs i'm already familiar with.

* **Case  7**: As a kubernetes admin I want to use the same provisioner for multiple clases of storage without re-writing the provisioner.

* **Case  8**: ~~As a kubernetes admin I want the naming pattern of provisioned persistent volumes to follow a custom convention.~~ ***Note***: pv-name established by Dynamic Provisioning framework, no option to change.

* **Case  9**: As a storage admin I want to deliver provisioners to the kubernetes admin and provide updates/changes in as automated fashion as possible.

* **Case 10**: As a kubernetes administrator I want to secure the node performing the provisioning from the rest of the network.


# Design
The framework for provisioning has already been established.  This design covers a custom provisioner running on the established framework. The design presented covers all of the use cases above.

## Creation of Provisioner
The storage administrator is responsible for creating a provisioning container.  The details and contents of the provisioning container aren't important only that it creates a new Persistent Volume before exiting.  This will require *kubectl* within the provisioner and a service account token passed into the container.  

Its the storage admins job to provide the kubernetes adminstrator with an appropriate container and the kubernetes admins job to ensure the container is availble on the appropriate nodes.  Properties passed into the provisioner could be on a secrets volume or as environment variables which can then be parsed by shell scripts or binaries inside the container.  

To support Use Case #10, the flex provisioner may accept a node selector to target specific nodes for provisioning.  If no node-selector is specified then any available node may be used.  Containers will run as pods.

## Provisioner Config
The kubernetes admin is responsible for configuring the provisioner.  Configuration for custom provisioners is similar to configuration of storage-claseses using a ConfigMap.  

A custom provisioners ConfigMap will require an additional field to provide a provisioning container which will perform the actual provisioning work.  Extra paramaters can be arbitrarily set by the kubernetes admin during configurion which will be passed to the provisioning container as well as additional properties provided by the API.  Such API provided properties include user, group, size, pv-name, storage-class and service account details.  The service account will be used to create the PV by the provisioning container.

**Example Config**: 
Here is an example markup file that the kubernetes administrator would use to configure a provisioner:

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
  service-toekn:23092390532lksf90kwelklfw
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
After the provisioner terminates it must return error or success.  A success return indicates that the new persistent volume has been created.  Its up to the provisioning framework to handle a provisioning success or failure.

## Sequence of Events
     
***Call Flow***: User submit PVC-->apiserver-->Dynamic Provisioning Framework --> Flex Provisioner --> Provisioning Pod-->Storage Control Plane

***Return Flow***: Storage Control Plane-->Provisioning Pod--(PV created via kubectl)--> api-server

Once the PV is created the PVC will attach by existing mechanism completing the flow.


# Function Examples

**Storage Admin** - Creating a Custom Provisioner:

Dockerfile:

``` Dockerfile
FROM centos
RUN yum update ; yum install iscsi-initiator-utils
RUN curl www.examples.com/me/iscsi-provision.sh >> /root/iscsi-provision.sh
ENTRYPOINT /root/iscsi-provision.sh
```
docker built . -t custom/iscsi-provision:latest
 
**Kubernetes Admin** - Configuring a custom provisioner for a silver storage-class:
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
  service-toekn:23092390532lksf90kwelklfw
  use-nfs-server:10.10.2.4
  zone:west
```

**Kubernetes User** - Same workflow as standard provisioner.
# Design Alternatives
**Alternative 1**: Use a binary/shell script on the master instead of a POD.  This violates Use Case  9 and possibly Use Case 5.  It also significantly complicates the passing of properties from kubernets API to provisioners.



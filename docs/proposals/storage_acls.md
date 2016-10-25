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

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

Today in kubernetes, the limits around storage assets is fairly limited and does not provide a means to restrict per namespace the type
of stoarge a user may request or limits around each request. This abstract proposes to provide ACLs for storage through the use of Quotas and 
Limits.

## Motivation

The traditional method for storage security, outside kubernetes, provides granularity down to specifc users and groups
in a system and the permissions on the filesystem at directory and file levels. By restircting the storage consumption first at the namespace,
storage can be allocated to specific groups and users in each namespace. In addition, the flexibility of this design can support both
shared storage and block storage use cases.

## Design
Defining how we access and use storage in Kubernetes is no different. And in some ways is more complicated. 
For simplicity we will distill storage security into two separate categories: access and use.

# Access

Access defines the authorization today done by the system to allow a user to authenticate and 
therefore have access to specific features in the system. This process today is not necessarily tailored to storage security. 
It is the outer layer of the onion that allows an admin to create users and service accounts to grant privileges. 
It does not consider storage specific security features when it does this validation. 


# Use Cases for ‘ACCESS’

Use Case: Administrator wishes to create user and permit the use of certain storage classes
Use Case: Administrator wishes to create group and permit the use of storage classes
Use Case: Cluster Admin wishes to allocate role “Storage Admin” to non-cluster Administrator for the purposes of creating storage

## Use

Use is the actual act of leveraging the storage feature once it’s enabled in your namespace. Use will be restricted by two methods. The SCC (Security Context Constraint) which as it is merged upstream will become the PSP (Pod Security Policy) and the use of ACLs controlled by Quota and Limits. 
Kubernetes controls are managed differently than traditional storage in that we can restrict down to a file level if necessary. Though, it embraces the method of hierarchical permissions, it depends on administrators management of first and foremost, namespaces, to restrict users and functions. Therefore, the need to provide the same file system level security is eliminated by only granting users specific functions in that namespace.

# Access to Physical Storage
There are features of storage access we do wish to control at a more granular level. Today three characteristics of the file systems that controls user access:

FSGroups: This is the filesystem group permissions for block storage. The Storag or Cluster administrator is responsible for allocating the GID for the storage asset at creation
SELinux
User specific SELinux parameters are not currently supported with the ability to apply SELinux labels after provisioning storage
Supplemental Groups: This is the shared storage group permissions for the storage asset. These must match what is already created on the physical asset in order for the pod to access the storage
None of these parameters should be requested by the user...this should happen ABOVE the request for storage
Range should not be so widely defined to enable users to access storage outside their circle

In addition to applying GIDs some storage plugins require credentials for certain administrative functions.

Access to Storage Features
Once the user is restricted based on the methods provided in “ACCESS” we want to further limit access to certain features. This has two distinct pieces. We want to limit:
What a user is allowed to do:

Quotas: provides constraints that limit aggregate resource consumption per project
Persistent Volume Claims

Ability to request specific storage 
  Storage Selectors

Ability to provision storage dynamically:
  Storage Classes
Selectors

How much storage the user may consume: 
  Limits: enumerates resource constraints per project 
    Size of request
    Enforcement by Quota that calculates usage
    Number of requests/claims


This PR will focus on providing a means  of Access Control Lists (ACLs) based on leveraging and extending Resource Quotas:
'''
apiVersion: v1
kind: ResourceQuota
namespace: $the_users_namespace
metadata:
  name: storage-resource
spec:
  hard:
    storageClass:
      Gold:
        requests.storage: 1
        limits.storage: 2Gi
      Silver:
        requests.storage: 2
        limits.storage: 2Gi
      nil:
        requests.storage: 2
        limits.storage: 4Gi

apiVersion: v1
kind: VolumeSecurityPolicy
namespace: $the_users_namespace
metadata:
  name: my-vsp
spec:
  StorageClass:
    - Gold
    - Silver
    - nil
'''

Use Cases for ‘USE’:

Use Case: Administrator wishes to create non-default provisioner and restrict it’s use to a specific namespace
  Default behavior: Disabled for ALL users
Use Case: Administrator wishes to create DEFAULT provisioner 
  Default behavior: If SINGLE cloud provider is defined, default class will be created with these credentials
  Default behavior: Enabled for ALL users
  Controls: # of claims restricted by limits/quotas
  Special considerations:
    What do we do when there is more than one cloud provider defined?
    What do we do when there is NO cloud provider defined

Use Case: Administrator wishes to delete provisioner
  Error message should be returned to user that provisioner unavailable. 
  Pending claims should be rejected with same message
What happens if Administrator deleted default?
  Should default to storage class ‘nil’
Use Case: Administrator wishes to change a parameter of the provisioner with minimal disruption
  Possible parameters: credentials, allowed users
Use Case: Administrator wishes to limit the requests of a storage class based on namespace
Use Case: Administrator wishes to restrict storage allocation for a given storage class based on namespace 
Use Case: Administrator wishes to restrict requests (claims) for a stoage based on namespace
Use Case: User wishes to remove claim to storage triggering a ‘detach’ 
Use Case: Administrator updates GID in SCC for block storage to new GID, access to the storage should continue to work with the current pod spec 
Use Case: PV having GID that isn’t part of the range defined in the SCC and binding to a PV but not usable by the pod (Important customer use case this doesn’t address today)

Special consdierations for ‘USE’
Augment Quotas to include Storage with sub-classification for Storage Classes
Update default SCC to include binary value to enable/disable Storage Selectors
Augment Limits to include storage with sub-limits per Storage Class
Create Storage UI with Storage Classes as primary method for claims
Extend Storage UI to provide ‘advanced’ section for selectors only
 Special considerations for for ‘USE’
By using quotas as namespaced attribute it assumes all users in a namespace have equal rights to storage
Default disabling of Selectors will discourage the user of this ‘special case’ to train users to leverage Storage Classes
We will only provide ACLs around Storage Class so if a selector is enabled, users can see all PVs with labels
Responsibility to managing these effectively will fall to the administrator
Scaling may be affected by environments that need several different types of storage classes
Ideally, administrative consoles should provide a way of managing storage classes and seeing where they are allocated
Quotas are typically on static resources: Adding Storage Classes could potentially be disruptive due to shear number and changes to them
Restarts / Caching etc... 
Storage should be considered a ‘precious’ resource and limited by default
This is currently the opposite behavior for Quotas




The design of storage ACLs hinges on the administrators adoption of Storage Classes as a secure way to present storage options to it's users. 

The design should consider the following use cases:

1.  As an Administrator, I want to create a storage class 'gold' that dynamically provisions expensive storage that I only want specific users to have access to.
    
2.  As a user I want to be able to see a catalog of storage assets for my consumption that provide a level of service and/or geography so I don't have to understand the specifics of the storage asset

3.  As a user I may need direct access to the storage asset outside the application (ie. SSH). I should be required and able
    to store a user specific secret for such access.
    
4.  As an Administrator, I should be able to update my credentials as necessary with minimal impact to users.


## Alternate Design consideration

https://github.com/kubernetes/kubernetes/pull/35538

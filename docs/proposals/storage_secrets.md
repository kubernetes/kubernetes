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

In Kubernetes, the API object 'secrets' provide a means to store confidential credentials in a volume to be accessed by 
the Kuberenetes Master as well as other external resources for the pod, such as database login information. 

Creating a secret volume for the storage layer has use cases outside the norm of traditional access.  Today, the
access to the secret is typically contrainted to a namepsace and application within that namespace requiring the information.
Secrets are created for a purpose that can be isolated by one function alone. Storage security presents the obstacle that secrets 
with actual storage assets may not have a specific namespace when the volumes are created. Therefore, facilitiating the need for an
alternate approach.

## Motivation

Today, ACLs are not well defined in storage. An admin cannot create precise contraints for a user or group based on the actual storage
assets and have it flow through the system. In addition, administrators whom pre-create volumes to expedite the process or to limit users, don't know which user or group may want to request that resource. Since secrets today are namespace constrainted, we cannot attach the
secret globally. Nor, would that solution necessarily be desireable considering the possibility of exposure.

There are specific functions, unique to storage that require the credentials to perform different actions. Today, when performing an 
attach/detatch the node must have this information to perform this action. This would entail copying the secret information from the 
master to all the nodes in the cluster. Also, not an ideal configuration.

The design should consider, but not be limited, by how roles are defined and leveraged as part of the PSP. An alternate proposal will
consider the creation of a 'storage administrator' in Kubernetes to better define the storage creation workflow.

The design should also consider the need for provisioners defined within Storage Classes to storage credentials for access to both cloud
and on-prem resources.

## Design

This design should create a way to allow a consistent way to create storage secrets that don't allow the user to access to administrative
functions.  The design should allow different secrets to be stored for specific function: user secret, disk encryption secret, administrative
secrets.

The design should consider the following use cases:

1.  As an Administrator, I want to store secret credentials to allow creation and mounting of externally defined storage resources. I
    I do not want to allow the consumer of the storage access to my credentials as the security level between users may be different.
    
2.  As a user I should be able to create and remove requests for storage through the use of a persistent volume claim or by leveraging
    an administrator defined Storage Class. I should not have to provide seperate credentials to create these assets. 
    The binding/unbinding to the assets should be opaque to the user.

3.  As a user I may need direct access to the storage asset outside the application (ie. SSH). I should be required and able
    to store a user specific secret for such access.
    
4.  As an Administrator, I should be able to update my credentials as necessary with minimal impact to users.


## Alternate Design consideration

It has been proposed in the community to append the namespace to a secret. This solution would allow the user(s) within that namespace to 
leverage the admin's credentials to perform a bind/unbind of the storage resource.
Reservations about the given proposal:

1.  The user could use the credentials for a purpose outside attach/detach by specifying the secret name from the pod spec.

2. The user could decode the secret and update the credentials affecting other users.

3. Without the precense of ACLs for storage classes, any user in any namespace can claim to a volume and obtain the administrator
   credentials.
   
4. Allowing the user access to administrative credentials/secret can present unauthorized access and prevent audit of the user as the 
   credentials are being shared.

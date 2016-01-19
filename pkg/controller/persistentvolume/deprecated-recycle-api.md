## Deprecation of persistentvolume recycler controller behavior
The function of the persistent volume recycler will change in future kubernetes releases.  The current implementation will remain and function as is for the short term but expected to be removed or replaced in future releases.  

## Summary
The current recycler design only allows for user space volume erasure operations.  This is inadequate for control plane or scrubbing that requires elevated privileges.  Future redesign is anticipated and this document serves as warning of future API change. **USE RECYCLE API AT YOUR OWN RISK AS IT MAY BE REMOVED OR SIGNIFICANTLY CHANGED IN FUTURE KUBERNETES RELEASES**.

## Rational
With the inclusion of configurable dynamic provisioning and storage service classes its expected that most storage volumes will be created dynamically.  In situations where the volumes can be created dynamically its more reliable and secure to "recycle" by deleting a volume and recreating it.

Additionally most filesystems do not remove block data when a user space delete operation occurs.  The next user of a recycled volume could gain access to the previous users data.

In the short term the user space volume scrub is useful and will remain for backwards compatibility.  As the dynamic provisioning feature is finished the future direction of the recycler will be discussed and designed during storage SIG community meetings.  The new design may include leaving the recycler concept with an enhanced implementation or removing it totally.  The decision will be based on community feedback.

## Deprecated Design
As of Kubernetes 1.2 if a volume is marked for recycle when it is deleted a special recycle container is attached and 'rm -rf's the data on the volume.  This behavior is expected to change or be removed.


## Timeline

Kube 1.2 : Notice of deprecation
Kube 1.3 : Community discussion and direction of Recycler future API
Kube 1.2 + 1 year : Remove old recycler API

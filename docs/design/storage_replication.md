# Objective 

The purpose of this document is to gather ideas and function for the use of storage volumes with replication controllers.  Once the requirements are established a code design will be added.  

The current design allows only one volume to map to a replicated pod. Its desirable when scaling an application that each replica receives its own unique volume.

# Goals

* Determine how a replication controller should map replicas to volumes
* Determine how volumes are handled when an RC scales up or down.
* Define the code design and markup changes required for Replication Controllers to assign unique volumes to each replica

# Assigning Volumes to Replicas
There are two potential volume type assignments for replication controllers- existing unclaimed volumes and non-existant dynamically provisioned volumes.

## Existing unclaimed volumes
Existing unclaimed volumes are persistent volumes pre-configured by an administrator and claimable by name to an end user.  

The user must provide a list of PVCs as part of the replication controller pod definition which can be claimed by each replica.  Replicas can't exceed the number of volumes listed unless a dynamic provisioning label is provided.  ((Another option to a list of PVCs is a user supplied regular expression for selection of pre-provisioned PVs)).

## Dynamically provisioned Volumes
A user would specify a provisioner label in the replicated pod definition and the replication controller would provision new volumes for each replica.  **Note:** The dynamic provisioning feature is still in alpha.  

## Order of replicas claiming volumes
Replicas will first get volumes from the list of available PVs then dynamically provision PVs as needed if a DP selector is supplied.

## Scaling up and down
* Scale up - When a replication controller is scaled up new PV are acquired first from the listed pool of predefined PVs.  Once the pool is empty (or if there is none) PVs are dynamically provisioned based on the provisioning selector.

* Scale down - When a replication controller is scaled down PODs release their persistent volume claim.  In the case that the PV was a pre-created PVC listed in the POD spec it may be reclaimed as such if the application is scaled back up.  In the case that the volume was dynamically provisioned, the provisioned volume will remain unclaimed and will not be re-claimed if the application is scaled back up.  In the case of dynamically provisioned volumes a new volume will be provisioned each time.

# Design

**TBD**

# Markup

**TBD**


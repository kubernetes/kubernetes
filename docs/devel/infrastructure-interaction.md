**Problem**

We've got at least 3 cases where kube needs to know about and interact with functionality provided by the underlying infrastructure. The 3 I can think of from the top of my head are:

1. minion discovery
2. storage (be it a GCE PD, NFS, Ceph, or something else [durable data is a little special])
3. externally reachable functionality (external IPs, external load balancers, ssl termination points, etc)

We don't seem to have a common pattern how kube and the cloud provider/infrastructure should be working together to expose functionality. I'd like to start a discussion of how I see at least a start of how external functionality and kube can get along.

**The three ways kube and infrastructure interact**

1. a new infrastructure ‘function’ is created and a kube resource is automatically created or kube automatically can use this thing (this is how minions work)
2. a new kube resource is created and the infrastructure satisfies it (this is how pods/service providers work)
3. a new infrastructure resource is created but kube does not know about it or make use of it until it is told (this is how GCE PDs work)

All three of these have a LOT of utility and value.  At the largest scales 1 and 2 seem likely the right way forward.  But, focussing on 3, right now, is the model that best fits most legacy enterprise workflows and seems to have the most clear way to work across differing infrastructure platforms.  I also believe that a clean separation between infrastructure functionality and kube resources will give us a mental model which makes building the automation toward 1 and 2 easier.  So this document focuses on number 3.  But we would be wise to think about the day when we can define a kube resource and have the infrastructure automatically create the underlying functionality and the day when we can create the functionality in the infrastructure and have it automatically available as a kube resource.

If we attack the problem right now, today, as problem #3 it also leaves the door open for people to write tools on top of kube and their infrastructure to mimic #1 and #2.  Which can hopefully help us one day bring those tools into kube.  But decoupling is a very important first step.

Between minions, GCE PDs, and GCE external load balancers, all of these have different implementations with little coherence for non-gce configurations. So thinking generally, how should kube and infrastructure functionality work.  If I’m looking at the kube cluster from that 3rd infrastructure interaction perspective (and possibly even the 1st perspective) I think we must view that underlying functionality as pets, not cattle.  That’s just a perspective I’m bringing into this discussion.

To decouple kube and the infrastructure I believe that our model of interaction should require three distinct steps and three distinct parties interacting.

**Design Idea**

There are three parties who need to operate on the system in my mind.  They may be actual people running things on a command line or automated tools doing something similar.  That’s not really relevant.

1. Infrastructure admin.  The person with the permission and ability to make changes to the infrastructure.
2. Kube admin.  The person with the permission to change how the kube cluster itself operates.
3. Kube user.  The person who just wants to write and run an app.

My actual design decoupling can be broken down into 3 steps.

1. The “infrastructure admin”, needs to make these external functions available to kube, potentially via a cloud provider interface similar to how minions are handled today.  We would want something similar for storage and external reachability functionality.  It seems likely that we may want to decompose the cloud_provider into multiple smaller providers.  ‘minion_provider’, ‘storage_provider’, etc.  But that is a specific detail to work out per infrastructure concept.

2. The “kube admin” needs to explicitly tell the cluster to use the functionality exposed by the infrastructure. In the case of minions this defaults to “*” but we have the –minion_regexp. That means minions sorta fit this model. GCE PDs fit as you have to specify the GCEPersistentDisk.  The specifics of how the kube admin tells the cluster to use the functionality will likely be functionality specific.  (Although I’ll give my thoughts on some of these things below)

3. “Something” needs to watch the cluster to see how the resource is supposed to be used and make that happen. Minions fall down here, it's more that the apiserver reaches out to them, but I think that’s an ok exception.  The model still mostly fits for GCE PDs. The kubelet watches the pod description and from the volume info in the pod definition it mounts and actually uses the storage. I think a different, but very similar, pattern should be generalized and should work well for externally reachable functionality (load balancers, external IP, etc) and other storage mechanisms.

I'll work through a couple of examples of what I'm thinking. Say an external load balancer (conceptually should map to external IPs or whatever else) and an NFS server (should map to other storage, including GCE PDs)

**External Load Balancer Example**

If I install an external load balancer into my datacenter with some form of reachability into the cluster (aka it can reach pods via pod IP addresses) I believe that the cloud provider interface needs to tell the cluster about the load balancer via some sort of “name” or “label” or whatever detail we agree upon. That's step 1. Step 2 is, in my mind, creating a new resource in kube. This resource should be a mapping between a service/pod/label and the 'name'/'label' of the external device. Yes, I’m talking about a first class resource like a pod, service, minion, etc. Step 3 is that the external load balancer (or a plugin in front of the load balancer) should watch the api server for new assignments and make the magic configuration changes for my load balancer.

**NFS Server Storage Example**

As an example lets say I want to use an NFS server to expose 10 exports for pod storage. I believe that either an admin or the infrastructure itself should make all of the information needed to mount those 10 exports available to the API server. I think this should be done as a new ‘volume’ resource (yes, very like the volume portion of the pod definition).  So (for NFS) the cloud provider (or maybe a kube admin) would make available the servername, export name, and any credentials necessary to use the mount. Now the kube admin/operator/user should be able to define a pod which references the storage by name. He should NOT have to know all of the storage details (as the volume requires today). The apiserver should take the volumeMount from the pod definition, the volume resource, and make all of that available to the kubelet. The dev who wrote the pod was told to use 'eparis.volume.42' and that's all they know. The kublet however gets nfs-server.example.com:/export/42/eparis.volume.stage/ along with a kerberos keytab from the apiserver. It can now mount and use the NFS mount.  I believe the concept should be able to map pretty cleanly to GCE PDs, or any storage concept (durable/empty/host are a little special, but not terribly outside the model.)

Openshift or others are certainly going to want to write a high level tool, which attempts to automatically create NFS exports or purchase more storage from GCE when requested. Thus imitating by doing work at a higher level the other infrastructure kube interaction perspective.  But the clean separation makes that possible rapidly.  That tool should interact initially with the infrastructure and the API server, to make the volume resource.  Although eventually I’m sure they’d like to choose to interact with one or the other.

Separating the volume to a first class resource instead of part of the pod definition gets you all sorts of cool stuff.  Couple of initial thoughts are that pod definitions become much more portable, since the infrastructure specific portions are abstracted out.  Access controls become much more reasonable, as you can easily do checks on creating the volume resource and between volume resources and pods.  More thoughts below...

Is this a sane design and starting point when talking about how to solve storage and external ips? If its a sane design is there interest in trying to normalize the minion stuff? Obviously actually solving any of those problems have 1,000,000 other details, but is this a sane design to move forward with?

**Design Restated**

1. infrastructure makes functionality available to kube
2. kube admin/user explicitly tells the system how to use those resources.
3. 'something' pulls that explicit information and tries to implement the request.

(man, that sounds exactly like the original kube design!)

**Mapping minions, GCE load balancer, and GCE PDs today with this perspective**

***Minions:*** minion discovery ideas are schizophrenic at best. We've got –machines and –minion_regexp. Which are mutually exclusive and obviously not coherent together. Trying to rationalize minions with this design proposal works great for the ‘real cloud provider model’ but it is admittedly not a nice admin experience without it.  I think we should get rid of --machines entirely.  To solve the admin experience we should plumb the ‘add a minion to etcd’ API through kubectl. I’m not quite sure how to make adding minions with kubectl work sanely with the regex and the list of minions from the cloud provider.  Maybe just disable it for certain cloud providers.  Maybe have “kubectl create/delete minion” merge with the regex.  Details.  Magic.

***GCE load balancer:*** In this case it’s handled by ‘createExternalLoadBalancer=true in the service file.  This causes the apiserver to make an (async) push type call into gce.service.ForwardingRules.Insert().  I seems to me that external services like this should be modeled as a new resource type separate from the service and that instead of the apiserver pushing the update to the cloudprovider/gce another task should run which pulls from the apiserver and does the push.  We’re trying to get away from the push model, right?  But I think the new kube resource argument is best fought in something like the external IPs PR.  This is a high level thought.

***GCE PDs:*** Here the full description of how to mount and use the disk is the volume in the pod definition [name, type, partition, RO].  It is not exposed and validated on input and anyone who can create any pod can use any storage in the GCE project/zone.  I think a way to rationalize this would be to move the kube volume concept out of the pod definition and into its own resource type.  After the GCE admin allocates, pays for, and formats a new GCE PD the kube admin would have to explicitly add them as a volume resource to the apiserver.  Eventually maybe having the GCE cloud provider inject them automatically.  To start having the kube storage admin do it makes the most sense to me. The pod definition would change to not need the ‘volume’.  Instead it would only need the ‘volumeMount’.

***Durable data:*** Obviously this maps pretty well to durable data since we already were leaning towards a new resource for that.  Making that the ‘volume’ resource seems doable extremely sane.

***Host data:*** This is really interesting.  Now the kube storage admin can declare what parts of the minion he wants to expose to pods.  It’s not user/app dev that gets to declare he gets to access the host files.  Pods can’t say “hostmount /etc/shadow:/etc/shadow”.  Only the kube storage admin can screw that up.  Pretty cool feature…

***Empty dir:*** well, who cares...

**What this means**

I think in all 3 cases what I’m asking for is probably an additional step for kube admins.  For ‘no cloud provider’ they have to manually add minions.  For ‘external load balancers’ they have to create some new resource separate from the service.  For storage they have to create a new volume resource instead of that being a part of the pod definition.

What this gets us is a way to abstract all of these things.  I can use the exact same pod definition and not care if I am on GCE and using PDs or in house and using Ceph.  That is the problem of whomever sets up the volume resource (be that an admin or the cloud provider).  That resource can change significantly without the pod changing at all.  In all of these cases my hope is that the person writing the Pod or the App can remain ignorant of the underlying infrastructure and the kube admin, rather than the kube user has to deal with those details.

Thoughts?

API Groups Design Proposal
--------------------------

## Purpose

Goals:

- document current API groups
- firmly decide what group Job goes into when it goes to stable.
- suggest (not binding) how other existing and future types might be (re)grouped
- get feedback on those suggestions

Non-goal:

- Figure out a plan for deprecating or compatibly moving v1 apis to new groups
- Figure out a plan for moving  `/api/v1` to to under `/apis`.

## Current State

At HEAD as of 25 Jan 2016, we have 5 API groups:

```
/api/v1
/apis/extensions/v1beta1
/apis/abac/v0
/apis/abac/v1beta1
/apis/authorization/v1beta1
/apis/componentconfig/v1alpha1/
```

Those api groups have nearly 50 "top-level" types: types that are
visible to users and which you can GET and/or POST by themselves
(not nested in some other type).  Subresources, unused objects, and component config
objects are not included.

```
/api/v1
	persistentVolume
	persistentVolumeClaim
	pod
	replicationController
	service
	serviceAccount
	endpoints
	node
	namespace
	event
	limitRange
	resourceQuota
	secret
	componentStatus
	configMap

/apis/extensions/v1beta1
	horizontalPodAutoscaler
	thirdPartyResource
	thirdPartyResourceData
	deployment
	daemonSet
	job
	ingress
	replicaSet

/apis/abac/v0
	policy // not currently available through API, but might be some day.

/apis/abac/v1beta1
	policy // not currently available through API, but might be some day.

/apis/authorization/v1beta1
	subjectAccessReview
	selfSubjectAccessReview
	localSubjectAccessReview

/apis/componentconfig/v1alpha1/
	kubeProxyConfiguration

/apis/metrics/v1alpha1
	rawNode
	rawPod
```

## Upcoming Additions

New top level objects being considered:
- `template` and `parameters` are added by #18215.  Other things do not
   depend directly on them.  They work as an independent template expansion
   service.
- `Node`, `Pod` metrics types are being added, tenatively to `metrics` group. (see #18824)
- `PetSet`: https://github.com/kubernetes/kubernetes/pull/18016 
- `heartbeat` by #14735
- `externalServiceProvider` by #13216
- `podSecurityPolicy` and maybe `PodSecurityPolicyBinding` by #7893 
- `workflow` by  #18827
- `scheduledJob` by #11980
- `metadataPolicy` by #18262
- `ingress` may change, and probably will move out of the main api group.
- `podStatusResult` should maybe move to a kubelet API group?

## Inter-Kind references

I list some dependencies here.

- `ScheduledJob` depends on `Workflow` and `Job`; `Workflow` depends on `Job`
- `Job`, `ReplicationController`, `ReplicaSet`, `Deployment` and `DaemonSet` make `Pods` via PodTemplates.
  - this is a strong dependency.
- Pods that use PersistentVolumeClaimVolumeSource depend on PersistentVolumeClaim object (`ClaimName`).
- PersistentVolumeClaims depend on PersistentVolume names.
- PersistentVolumes depend on PersistentVolumeClaims via ClaimRef.
- `GlusterfsVolumeSource` of a Pod depends on `Endpoints`.
- Pods depend on `Secrets` several ways: `SecretVolumeSource`, EnvVars.
- `Pods` depend on `ServiceAccounts` via `ServiceAccountName`.
  - this seems like a pretty strong dependency.
- `Pods` depend on `Nodes` in that `Pods` have a `NodeName`.
- `Pods` with an `RBDVolumeSource` or `CephFSVolumeSource` or `FlexVolumeSource` depend on `Secrets` to store a secret.
- `Pods` depend on `Secrets` for `imagePullSecrets`.
- `ServiceAccounts` refer to `Secrets`.
- `Pods` can depend on `ConfigMap` or `Secret` via env vars, etc.
- `ReplicationControllers` depend on `Pods` via a `PodTemplate`.

## Reasons to have more than 1 group

Why do we not put everything into 1 group.   That would be easier.
Well, clearly we need namespacing, and we cannot control the naming of
addons,

But, why not one group for "the project" and then add another group
for any "third-party" APIs?

The three most important reasons, according to @bgrant0607, are:

1. To allow api versioning at different rates
  - This is described in #635.
  - It is particularly for hosted implementations of Kubernetes.
2. So that users can focus on learning one group at a time.
3. Groups will eventually allow extension by third parties that we as a core project
  would not accept.  We as a core project have to use the extension mechanism to keep
  ourselves honest, though.

Other, unranked reasons:

- So that security/policy admins can audit/enable an API group at a time, and not need to
  review all kinds at once.
- The same reasons programs have packages: to enforce boundaries, and prevent coupling (taken from #635).
- To allow replacement of project-provided kinds with proprietary types
  - PaaSes on top of Kubernetes might to do this.
- So that different groups within the kubernetes project can have a greater degree of autonomy
  - Conways law 
  - have trusted api reviewers per group.

## Ways to Group things

Listed below are some ways that you might want to group things.
It is not a proposal to do all those things; in fact some are in conflict.

### Proposal from #635

Issue #635 proposed this grouping:

```
apis/podScheduling/v1
	Pods
	PodTemplates
	Bindings

apis/services/v1
	Services

apis/rcs/v1
	ReplicationController

apis/nodes/v1
	Nodes
```

### Proposal from #8190

```
# Does not suggest names for groups

# end-user execution objects:
	pod
	secret
 	configMap
	persistentVolumeClaim

# end-user networking-related objects:
	service
	ingress

# end-user app deployment objects:
	replicaSet
	deployment
	petSet		# added

# end-user batch objects:
	job
	scheduled job
	workflow 	# added

# app admin objects:
	namespace
	limitRange
	resourceQuota

# identity/auth objects:
	serviceAccount
	user	 (future)

# system / tooling objects:
	event
	endpoints
	componentStatus
	metrics

# infrastructure / cluster admin deployment objects:
	daemon set

# infrastructure / cluster admin infrastructure objects:
	node
	persistent volume

# experimental apis and plugins
	...
```

### Things nodes care about vs other things

One way to slice this is what the node (kubelet + kube-proxy) needs to understand. Those types are "core" in a way that others are not.

Things nodes care about:
- User execution concepts: Pod, Secret, ConfigMap, PersistentVolumeClaim
- Infrastructure concepts: Node, PersistentVolume
- Networking concepts: Service, Endpoints

### Admin vs User

Issue #3806 proposed splitting "Cluster" and
"User" resources.  Applied to the current state,
that would look something like this:

```
/api/v1
	persistentVolumeClaim
	pod
	replicationController
	service
	serviceAccount
	endpoints
	event
	secret
	petSet
	workflow
	scheduledJob
	componentStatus   # Maybe goes in apis/cluster?
	ingress           # Maybe goes in apis/cluster?
	podSecurityPolicyBinding

/apis/metrics/v1alpha1
	Pod

/apis/cluster/v1
	node
	namespace
	persistentVolume
	limitRange
	resourceQuota
	daemonSet
	metadataPolicy
	podSecurityPolicy

/apis/clustermetrics/v1alpha1
	Pod
	Node

/apis/extensions/v1beta1
	horizontalPodAutoscaler
	thirdPartyResource
	thirdPartyResourceData
	deployment
	job
	replicaSet
```

Evaluation:

- Everything still in the main API (except componentStatus and maybe podSecurityPolicy)
  is stuff that makes sense to namespace, since they are things you replicate if
  you make a "copy" of a service or system.
- Everything in the `apis/cluster/v1` group does not need a namespace,
  except limitRange, and resourceQuota
- Everything in `apis/cluster/v1` is ostensibly stuff that a tenant does not need
  to view or modify to use the system.  And it is stuff that the a hosted
  kubernetes provider certainly wants tight control over, and may want
  to hide so that it can be implemented differently.
- non-admins can focus their learning on the User group.

### Vertical Grouping

Group things that work with the same type of workloads.

Sketch:
```
/apis/batch/v1
	scheduledJob
	workflow
	job

/apis/longrunning/v1  # need better name
	replicaSet
	deployment
	petSet

/apis/cluster/v1
	daemonSet

# ReplicationController is deprecated by ReplicaSet.
```

Rationale for `batch` group is that if you are running a batch job, you are likely to care about
all three things (run-to-completion, start-at-a-time, and dependencies).

Rationale for `DaemonSet` being in separate group is that typically only the cluster admin
needs to deal with DaemonSets.

Other *verticals*, to use the Omega term, would be peers of these groupings.
For example, a map-reduce controller might have a `MapReduce` object that specifies
the data and program, and then it might make its own Pods.  How it negotiates
for resources is a little unclear, though.

### Horizontal Grouping

Group things that are complementary, and have a similar control loop and/or similar level of abstraction.
For example, put `Job`, `DaemonSet`, `ReplicaSet`, `Deployment` and `ReplicationController` all go into a group
because they all make Pods.


- pro: we have three similar things.  let us group them.  yay!
- con: what about `PetSet`?
- con: conflicts with *vertical* way of grouping things, and with *user vs admin*.

### Respecting Dependencies

If Object A1 in api group A strongly depends on Object B1 in api group B,
and if api group A is "optional" then B should be "optional".
But what does "strongly depend" mean?

If Object C1 and Object C2 have a circular dependency, then it suggets that they belong in the same
API group.

### Conclusion

The "user vs admin" has a clear benefit for hosting providers in that they can chose to not expose
a single group which contains a bunch of things they do not want to expose (a custom scheduler
might get nodes information from a different API).

The "Vertical" grouping seems like it fits well with "conways law" and that "vertical" is the pattern that
people who build large extensions will tend to follow.

Therefore, I suggest a combination of the "user vs admin" and the "vertical" groupings.

Sketch of final groupings (not showing types that may remain for deprecation purposes):

```
/api/v1
	persistentVolumeClaim
	pod
	service
	serviceAccount
	endpoints
	event
	secret

/apis/componentsStatus
	componentStatus

/apis/ingress
	ingress

/apis/metrics/v1alpha1
	Pod

/apis/cluster/v1
	node
	namespace
	persistentVolume
	limitRange
	resourceQuota
	daemonSet
	metadataPolicy
	podSecurityPolicy

/apis/scaling/v1
	horizontalPodAutoscaler

/apis/thirdParty
	thirdPartyResource
	thirdPartyResourceData

/apis/batch/v1
	scheduledJob
	workflow
	job

/apis/longrunning/v1  # need better name
	replicaSet
	deployment
	petSet
```


Suggested order of work:

- `Job` to move to `batch` for 1.2.
- `ScheduledJob` and `Workflow` to move to `batch` when the become `v1`.
- `DaemonSet` to move to `cluster/v1` when it graduates from v1.
- `podSecurityPolicy` to move into `cluster` prior to it becoming `v1`.
- no immediate plan to move other types out of their current locations.


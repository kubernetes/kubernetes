# Proposal: Solving the Confused Deputy Problem with Acting-As

## Proposal

**Ongoing Problem**: we have had confused deputy bugs in RC-manager, job-manager, and HPA-manager.
It is an insidious problem in general.  We need to attenuate the authority of controller-manager.

**Core Kubernetes Principle**: controllers from the core project, like replication controller, are
not special -- they are on a level playing field with user-implemented components.  (Only apiservers
are special, and they should do as little work as practical.)

**Observation**:  Instead of POSTing an RC and letting the RC-manager watch it for me, I could write a
go program that watches my pods, and makes new ones for me, emulating an RC.  Currently the auth
flow for those two cases is totally different.  But it should not be very different, (see above
Principle).  

This diagram shows how it works today when a user creates an RC, and then the RC-manager
creates a pod.

<img src="https://www.lucidchart.com/publicSegments/view/14d7b49f-469c-4bc4-8bf7-f13581ee0220/image.png"
alt="UML Sequence Diagram for Current Replication Controller">
<!-- Original document: https://www.lucidchart.com/documents/edit/34454705-cff6-4b42-a95f-24d69a36184c
     contact etune@google.com to request access -->

This diagram shows how it woud work today if a user created a program that was itself a Kubernetes
client, and acted as an RC, but for a fixed template, number of replicas, etc.

<img src="https://www.lucidchart.com/publicSegments/view/a719bd66-191f-478f-8c42-bc033d19ef25/image.png"
alt="UML Sequence Diagram for User-written Replication Controller">
<!-- Original document: https://www.lucidchart.com/documents/edit/92c9f481-a104-4283-8c8d-6bec71a1d9d7
     contact etune@google.com to request access -->

This diagram shows a proposal for how I think RC-manager should work.  It uses a new header (or
maybe query parameter?) which tells the apiserver that the RC-manager wants to Act-As
a different user. It is much closer to the "user-written replication manager" case.

<img src="https://www.lucidchart.com/publicSegments/view/c59b1964-2345-4910-8580-d4c636f28fa3/image.png"
alt="UML Sequence Diagram for Proposed RC-manager">
<!-- Original document: https://www.lucidchart.com/documents/edit/7c8cdcde-9803-4211-8cf1-a05c6da3fbc6
     contact etune@google.com to request access -->

Things to note in the new flow:

- controller-manager is using authorized port instead of the localhost port to talk to 
   kube-apiserver
- controller-manager sets a header to say it wants to lower its root-like powers in order
  to act as less powerful user (which solves the confused deputy).


## Argument for proposal

This change would protect us from the confused-deputy problem.  This is important because:

1. we have already had confused deputy bugs in many of our controllers.  there will be others.
1. this is a common class of bugs in systems such as this.
1. we might trust ourselves to keep controllers in the main repo free from these bugs;
   but we will not have time to closely review contributed code for these bugs.
1. out authorization rules will get more complex over time, so deputies will have
   to have new, more complex checks in them.  Bug fixing contrib code will be hard,
   since authorization is compiled into the controller.
1. Writing a controller in a language other than go difficult because you
   will not be able to reuse on-controller-auth code.

## Work items

The specific changes needed for this new flow are:

- change controller manager to use the secure port rather than 127.0.0.1 on the
  master This does require users to change their flags, but we can support the
  old way for several release cycles.
- change apiserver to recognize an "Acting-As" header in requests, and to check whether the
  originally authenticated user is able to act as that user.
- Record both originally authenticated user and acting-as user in the apiserver logs for
  auditability. (Bonus).
- change controller-manager to set "Acting-As" header in requests.  It will be hardcoded to use
  the default service account of that namespace when making a request.  

### Controllers other than replication controller

- Job manager: should change the same as replication controller
- DaemonSet manager: should change the same as replication controller
- Endpoint controller: not sure if needed to use "Acting-As".
- garbage collector: not sure if needed to use "Acting-As".
- node controller: not sure if needed to use "Acting-As".
- route controller: not sure if needed to use "Acting-As".
- deployment controller: should use "Acting-As" when CRUDing replication controllers. 
- service controller: does not create other object, so should not need to use "Acting-As".
- quota controller: not sure if needed to use "Acting-As".
- namespace controller: if it is executing initializers and finalizers, then it may need to "Acting-As" for some of those steps.  Needs thought.
- horizontal-scaler controller: should use "Acting-As" to ensure it does not scale things the HPA creator should not be able to scale.
- persistent-volume controller: not sure if needed to use "Acting-As".
- persistent-volume recycler: not sure if needed to use "Acting-As"
- serviceaccount.TokensController: probably should, since it CRUDs objects in response to changed made in other objects by users.
- serviceaccount.ServiceAccountController: probably should, since it CRUDs objects in response to changed made in other objects by users.

These do not all need to be convered in one go.  It should be possible to have an incremental transition.

## Alternatives considered

### Alternative: Access Review API

In this alternate design, Controllers would not use Acting-As header, and they would not have
authorization checks compiled into them.  Instead, they use an "Access Review" interface on the
APIserver that lets a controller ask "If I were user A would I be allowed to create this Pod?".

Problem with this is that a system admin has to fully trust any new controller (e.g. contributed
controller).

Recommendation: reject this alternative.

### Alternative: Special-purpose accounts

In this alternate design, the controller-manager authenticates
as a special-purpose account called "controller-manager-acting-in-namespace-foo".
No "acting-as" header.  Controller-manager actually stores credentials for all
the accounts it is allowed to act as, and when a namespace is created,
either the namespace admin consents to let controller-manager act in it,
or a namespace initializer process creates the account.

This is a good approach too.
It is analgous to creating a "bot" account on our github organization, and then adding that
bot account to the ACLs of our github repo.

Good things about this approach:

1. no special purpose header needed
1. no need to record two users in an audit log (original and acting-as).
1. no need for second authorization stage in the apiserver.
1. since the special-purpose account (e.g. "controller-manager-acting-in-namespace-foo")
   is added to the ACL for a particular namespace, a Policy Manager for that namespace
   could revoke that ACL if desired.

Concerns about this approach:

1. more complex to get right at first.
1. requires that users see a special purpose account added to the ACL of their namespace/cluster
   for each use-case.  
1. workflow for automating adding these to new namespaces could get complex.  Even more
   so when there are multiple roles in a namespace.
1. automation of the workflow for adding these special-purpose accounts to ACLs requires
   a standardized ACL API for kubernetes, which competes with the goal of pluggable
   authorization in kubernetes.
1. figuring out where these special accounts fit into existing accounts and auditing frameworks.

Recommendation: keep in mind as something we may later evolve to.

### Alternative: Oauth-like

With Oauth, a user delegates his own authority to access his personal data.
In a kubernetes cluster, multiple users have access to a namespace,
and multiple users may modify a pod.  And, in the future, those users
may have different permissions within the namespace.  And one of those
users might leave the company, but the replication controller needs
to keep working.  So, traditional oauth flows do not seem like a
good fit.  Experience of my colleagues is that ACLs are better.
Products like Amazon IAM notably do not use oauth as a core part
of their solution.

Recommendation: reject this alternative.

### Alternative: Acting-as the human who create the RC

In this alternative, when the RC is created, the creating human user
is recorded in either the metadata of the RC or in the spec
of the RC.  When the RC-manager creates a pod, it acts as
that human user (or some intersection of the human and other
things).

Consider what happens if a human user to creates a replication controller
for an important service, and then leave the company.
The employee no longer appears in the user database, and so the apiserver
can no longer recognize that human user, and so it will refuse to
allow the RC-manager to act as that user.   Or the user still appears in the user
database, but was removed from all ACLs on termination.  So, again
the RC-manager cannot create pods.

Recommendatin: reject this alternative.

## Follow on work

You can safely skip this section, unless you totally understood the
above and have concerns about steps after that.

### Admission Controllers as confused deputies

Right now, admission controllers modify parameters of Pods, such as
setting a service account on the pod or (in the future) setting
the security context on the pod.  We think we want people to be able to
write their own admission controller plugins.  But the plugins are
acting as a superuser currently, and so have the confused deputy problem.
Also, the order of the controllers is quite sensitive.

I have ideas on fixing this, for a separate proposal.


### Multiple roles in a namespace
This proposal "solves" the confused deputy problem in the case where:

- the controller acts as the default service account.
- the authorization and admission control rules are the same within a namespace, regardless of who
  creates stuff (etune vs controller-manager acting-as default-service-account).

But, things will need to get more complex.  In partucular we will grow to have cases where we want
different roles within a namespace (role A vs role B) and when a principal who as permissions of
role A creates an RC, we want the RC-manager to also act in with role A permissions.  And when a
person with permissions of role B creates an RC, the RC-manager needs to act with role B.  I have a
sketch of how this will work, which I will present in a future proposal.

I have ideas on fixing this, for a separate proposal.

### Controllers have world read permission

Controllers need to have read permissions on their corresponding objects
in all namespaces in order to see when an object is created.
Could this be approved?  And do broad read rights need to be granted
to external/3rd-party controllers?

More on this later.  But this is the least pressing of all the problems,
I think.

# Cluster Upgrades

This is a position statement on Cluster Upgrades.  Submitted for feedback.

## Assumptions

Stable API types (v1, v2, etc) will to supported for fairly long times (e.g. 1 year, and probably longer).  This doc is not about that.

Critical security patches will be backported to all recetnt minor releases (maybe 1 year?  TBD).  This doc is not about that.

It is about "full upgrades" of the programs that make up the cluster (docker, kubelet, apiserver, controller-manager, addons, and their respective versions and flags and many options and input files).
in order to get new features and performance improvements.

User attitudes to upgrades differ.  Here are three user archetypes:
  - "yearly upgrader": In my company, we only upgrade software once per year, because that is how we have always done it.  Unless there is a critical security fix.
  - "eager upgrader": I want to upgrade my cluster to every new kubernetes release.  I enjoy reading the release notes for each new release of kubernetes with a cup of tea and a chocolate croissant.  Then I try out any interesting new apiserver flags before lunch.
  - "hosted upgrader": I use a hosted Kubernetes (e.g. GKE) and trust my hosting provider to keep my cluster up to date.  I just want it to keep working without any effort.  I do not know or care what flags my apiserver has.
We need to have a story for all these types of users.

Kubernetes project wants to release often (~4 releases per year currently, and many people would like to get to 6 or 12).  In each release, these things might change:
  - version of kubelet required by control plane
  - version of docker required by kubelet
  - apiserver, controller-manager, kubelet flags
  - storage representation of API objects
  - kernel version and base OS config requirements
  - existence of added fields on API objects (not captured by apiVersion) depended on by controllers, addons, etc.

It is practical for the Kubernetes project to test forwards and backwards compatibility for 

## Approaches to Upgrades

This lists some ways we could tell our users to do upgrades.

### Single-minor-version supported

For an upgrade of one minor version number (e.g. 1.5 to 1.6), Kubernetes project can:
 - provide release notes that explain requirements for upgrade
 - provide forward/backward testing
 - scripts to auotmate upgrade (maybe).
 - e2e tests that ensure upgrade is possible.

If Kubernetes only provides "single minor version" upgrades, then:
  - the "yearly upgrader" customer is unhappy, because she needs to do 4 or more consecutive upgrade steps (since a jump from say 1.5 to 1.9 is unlikely to work).
  - the "eager upgrader" and "hosted upgrader" are happy with this.
  - the kubernetes team is happy because the upgrade testing problem is tractible

### Multiple-minor-version supported

If kubernetes team has to support multiple minor version upgrades, then the testing matrix gets very large.  Ick.

### Cluster-swap upgrades

To upgrade your cluster from version 1.x to 1.y, you create a new, empty cluster with version 1.y.  Then you gradually bring up your services in the new cluster, and bring them down in the old cluster.


This approach does require application owners to stop and restart their pods/services.
This might not be as bad as it sounds since:
  - Kubernetes has a goal to make it very easy to turn up new clusters, so this should get easier over time.
  - Serious users will probably want to "qualify" new cluster software before using it in "production", so they will already be creating a "test" cluster with a new version before upgrading production.
  - Serious users will run "test", "staging" and "production" instances of their Deployments and services.  So, they will have experience with dealing with multiple instances of services.

## Proposal

1. Kubernetes project supports single-minor version upgrades, with forward/backward compat testing.
1. Kubernetes project may move to faster release cadence in future, and still only support single-minor-version
1. We recommend "Cluster Swap Upgrades" for "yearly upgraders", or for "eager upgraders" that got behind.
1. We recommend "Single-minor-version" upgrades for "eager upgraders" and "hosted upgraders".
1. We work to remove friction from the "Cluster Swap Upgrades" path.


## Details of Cluter Swap Upgrades

### Plain version

  1. start with cluster A that is 1.x
  2. create cluster B with 1.y, using kube-up
  3. kubectl delete object an in cluster A and kubectl create same object in cluster B.
  4. repeat previous step until done.
  5. when cluster A is empty, tear it down.

This approach can work well for "yearly upgrader", since x and y can be arbitrarily far apart (as long as the API versions are still supported, which they should be since we support these for a long time).
It may not be as good a fit for "eager upgrader", since it is a more heavyweight process just to try new features out.
It is not a good fit for the "hosted upgrader" who does not want to do any work for upgrades.

Problems:
  - Tear down and build up needs to respect dependencies between services, and without ubernetes, cross-cluster deps do not work.
  - May need twice as many machines in the worst case, unless there is a rescheduled to squeeze machines.

### Cluster-swap upgrades with Ubernetes

To upgrade your cluster from version 1.x to 1.y:
  1. start an ubernetes federation apiserver "F"
  1. put cluster A, that is 1.x, behind "F".
  1. start your services and deployments.
  1. create cluster B with 1.y, using kube-up
  1. put B behind F
  1. scale down nodes of A and up B
  1. stuff automatically shifts from A to B
  1. delete A when nothing is left on it.


Problems:
  - May need twice as many machines in the worst case, unless there is a rescheduled to squeeze machines.
  - A lot going on.

<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Deployments

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Deployments](#deployments)
  - [What is a _Deployment_?](#what-is-a-deployment)
  - [Enabling Deployments on kubernetes cluster](#enabling-deployments-on-kubernetes-cluster)
  - [Creating a Deployment](#creating-a-deployment)
  - [Updating a Deployment](#updating-a-deployment)
    - [Multiple Updates](#multiple-updates)
  - [Writing a Deployment Spec](#writing-a-deployment-spec)
    - [Pod Template](#pod-template)
    - [Replicas](#replicas)
    - [Selector](#selector)
    - [Unique Label Key](#unique-label-key)
    - [Strategy](#strategy)
      - [Recreate Deployment](#recreate-deployment)
      - [Rolling Update Deployment](#rolling-update-deployment)
        - [Max Unavailable](#max-unavailable)
        - [Max Surge](#max-surge)
        - [Min Ready Seconds](#min-ready-seconds)
  - [Alternative to Deployments](#alternative-to-deployments)
    - [kubectl rolling update](#kubectl-rolling-update)

<!-- END MUNGE: GENERATED_TOC -->

## What is a _Deployment_?

A _Deployment_ provides declarative update for Pods and ReplicationControllers.
Users describe the desired state in deployment object and deployment
controller changes the actual state to that at a controlled rate.
Users can define deployments to create new resources, or replace existing ones
by new ones.

A typical use case is:
* Create a deployment to bring up a replication controller and pods.
* Later, update that deployment to recreate the pods (for ex: to use a new image).

## Enabling Deployments on kubernetes cluster

Deployments is part of the [`extensions` API Group](../api.md#api-groups) and is not enabled by default.
Set `--runtime-config=extensions/v1beta1/deployments=true` on API server to
enable it.
This can be achieved by exporting `ENABLE_DEPLOYMENTS=true` before running
`kube-up.sh` script on GCE.

Note that Deployment objects effectively have [API version
`v1alpha1`](../api.md#api-versioning).
Alpha objects may change or even be discontinued in future software releases.
However, due to to a known issue, they will appear as API version `v1beta1` if
enabled.

## Creating a Deployment

Here is an example Deployment. It creates a replication controller to
bring up 3 nginx pods.

<!-- BEGIN MUNGE: EXAMPLE nginx-deployment.yaml -->

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.7.9
        ports:
        - containerPort: 80
```

[Download example](nginx-deployment.yaml?raw=true)
<!-- END MUNGE: EXAMPLE nginx-deployment.yaml -->

Run the example by downloading the example file and then running this command:

```console
$ kubectl create -f docs/user-guide/nginx-deployment.yaml
deployment "nginx-deployment" created
```

Running a get immediately will give:

```console
$ kubectl get deployments
NAME               UPDATEDREPLICAS   AGE
nginx-deployment   0/3               8s
```

This indicates that deployment is trying to update 3 replicas. It has not
updated any one of those yet.

Running a get again after a minute, will give:

```console
$ kubectl get deployments
NAME               UPDATEDREPLICAS   AGE
nginx-deployment   3/3               1m
```

This indicates that deployent has created all the 3 replicas.
Running ```kubectl get rc``` and ```kubectl get pods``` will show the replication controller (RC) and pods created.

```console
$ kubectl get rc
CONTROLLER                      CONTAINER(S)   IMAGE(S)      SELECTOR                                                        REPLICAS   AGE
REPLICAS   AGE
deploymentrc-1975012602         nginx          nginx:1.7.9   deployment.kubernetes.io/podTemplateHash=1975012602,app=nginx   3          2m
```

```console
$ kubectl get pods
NAME                            READY          STATUS        RESTARTS       AGE
deploymentrc-1975012602-4f2tb   1/1            Running       0              1m
deploymentrc-1975012602-j975u   1/1            Running       0              1m
deploymentrc-1975012602-uashb   1/1            Running       0              1m
```

The created RC will ensure that there are 3 nginx pods at all time.

## Updating a Deployment

Lets say, now we want to update the nginx pods to start using nginx:1.9.1 image
instead of nginx:1.7.9.
For this, we update our deployment to be as follows:

<!-- BEGIN MUNGE: EXAMPLE new-nginx-deployment.yaml -->

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  replicas: 3
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.9.1
        ports:
        - containerPort: 80
```

[Download example](new-nginx-deployment.yaml?raw=true)
<!-- END MUNGE: EXAMPLE new-nginx-deployment.yaml -->


```console
$ kubectl apply -f docs/user-guide/new-nginx-deployment.yaml
deployment "nginx-deployment" configured
```

Running a get immediately will still give:

```console
$ kubectl get deployments
NAME               UPDATEDREPLICAS   AGE
nginx-deployment   3/3               8s
```

This indicates that deployment status has not been updated yet (it is still
showing old status).
Running a get again after a minute, will give:

```console
$ kubectl get deployments
NAME               UPDATEDREPLICAS   AGE
nginx-deployment   1/3               1m
```

This indicates that deployment has updated one of the three pods that it needs
to update.
Eventually, it will get around to updating all the pods.

```console
$ kubectl get deployments
NAME               UPDATEDREPLICAS   AGE
nginx-deployment   3/3               3m
```

We can run ```kubectl get rc``` to see that deployment updated the pods by creating a new RC
which it scaled up to 3 and scaled down the old RC to 0.

```console
kubectl get rc
CONTROLLER                CONTAINER(S)   IMAGE(S)      SELECTOR                                                         REPLICAS   AGE
deploymentrc-1562004724   nginx          nginx:1.9.1   deployment.kubernetes.io/podTemplateHash=1562004724,app=nginx   3          5m
deploymentrc-1975012602   nginx          nginx:1.7.9   deployment.kubernetes.io/podTemplateHash=1975012602,app=nginx   0          7m
```

Running get pods, will only show the new pods.

```console
kubectl get pods
NAME                            READY     STATUS    RESTARTS   AGE
deploymentrc-1562004724-0tgk5   1/1       Running   0          9m
deploymentrc-1562004724-1rkfl   1/1       Running   0          8m
deploymentrc-1562004724-6v702   1/1       Running   0          8m
```

Next time we want to update pods, we can just update the deployment again.

Deployment ensures that not all pods are down while they are being updated. By
default, it ensures that minimum of 1 less than the desired number of pods are
up. For example, if you look at the above deployment closely, you will see that
it first created a new pod, then deleted some old pods and created new ones. It
does not kill old pods until a sufficient number of new pods have come up.

```console
$ kubectl describe deployments
Name:                           nginx-deployment
Namespace:                      default
CreationTimestamp:              Thu, 22 Oct 2015 17:58:49 -0700
Labels:                         app=nginx-deployment
Selector:                       app=nginx
Replicas:                       3 updated / 3 total
StrategyType:                   RollingUpdate
RollingUpdateStrategy:          1 max unavailable, 1 max surge, 0 min ready seconds
OldReplicationControllers:      deploymentrc-1562004724 (3/3 replicas created)
NewReplicationController:       <none>
Events:
  FirstSeen     LastSeen        Count   From                            SubobjectPath   Reason          Message
  ─────────     ────────        ─────   ────                            ─────────────   ──────          ───────
  10m           10m             1       {deployment-controller }                        ScalingRC       Scaled up rc deploymentrc-1975012602 to 3
  2m            2m              1       {deployment-controller }                        ScalingRC       Scaled up rc deploymentrc-1562004724 to 1
  2m            2m              1       {deployment-controller }                        ScalingRC       Scaled down rc deploymentrc-1975012602 to 1
  1m            1m              1       {deployment-controller }                        ScalingRC       Scaled up rc deploymentrc-1562004724 to 3
  1m            1m              1       {deployment-controller }                        ScalingRC       Scaled down rc deploymentrc-1975012602 to 0
```

Here we see that when we first created the deployment, it created an RC and scaled it up to 3 replicas directly.
When we updated the deployment, it created a new RC and scaled it up to 1 and then scaled down the old RC by 1, so that at least 2 pods were available at all times.
It then scaled up the new RC to 3 and when those pods were ready, it scaled down the old RC to 0.

### Multiple Updates

Each time a new deployment object is observed, a replication controller is
created to bring up the desired pods if there is no existing RC doing so.
Existing RCs controlling pods whose labels match `.spec.selector` but the
template does not match `.spec.template` are scaled down.
Eventually, the new RC will be scaled to `.spec.replicas` and all old RCs will
be scaled to 0.
If the user updates the deployment while an existing deployment was in progress,
deployment will create a new RC as per the update and start scaling that up and
will roll the RC that it was scaling up before in its list of old RCs and will
start scaling it down.
For example: If user creates a deployment to create 5 replicas of nginx:1.7.9.
But then updates the deployment to create 5 replicas of nging:1.9.1, when only 3
replicas of nginx:1.7.9 had been created, then deployment will immediately start
killing the 3 nginx:1.7.9 pods that it had created and will start creating
nginx:1.9.1 pods. It will not wait for 5 replicas of nginx:1.7.9 to be created
before changing course.

## Writing a Deployment Spec

As with all other Kubernetes configs, a Deployment needs `apiVersion`, `kind`, and
`metadata` fields.  For general information about working with config files,
see [here](deploying-applications.md), [here](configuring-containers.md), and [here](working-with-resources.md).

A Deployment also needs a [`.spec` section](../devel/api-conventions.md#spec-and-status).

### Pod Template

The `.spec.template` is the only required field of the `.spec`.

The `.spec.template` is a [pod template](replication-controller.md#pod-template).  It has exactly
the same schema as a [pod](pods.md), except it is nested and does not have an
`apiVersion` or `kind`.

### Replicas

`.spec.replicas` is an optional field that specifies the number of desired pods. Defaults
to 1.

### Selector

`.spec.selector` is an optional field that specifies label selectors for pods
targeted by this deployment. Deployment kills some of these pods, if their
template is different than `.spec.template` or if the total number of such pods
exceeds `.spec.replicas`. It will bring up new pods with `.spec.template` if
number of pods are less than the desired number.

### Unique Label Key

`.spec.uniqueLabelKey` is an optional field specifying key of the selector that
is added to existing RCs (and label key that is added to its pods) to prevent
the existing RCs to select new pods (and old pods being selected by new RC).
Users can set this to an empty string to indicate that the system should
not add any selector and label. If unspecified, system uses
"deployment.kubernetes.io/podTemplateHash".
Value of this key is hash of `.spec.template`.
No label is added if this is set to empty string.

### Strategy

`.spec.strategy` specifies the strategy to replace old pods by new ones.
`.spec.strategy.type` can be "Recreate" or "RollingUpdate". "RollingUpdate" is
the default value.

#### Recreate Deployment

All existing pods are killed before new ones are created when
`.spec.strategy.type==Recreate`.
Note: This is not implemented yet.

#### Rolling Update Deployment

Deployment updates pods in a [rolling update][update-demo/] fashion
when `.spec.strategy.type==RollingUpdate`.
Users can specify `maxUnavailable`, `maxSurge` and `minReadySeconds` to control
the rolling update process.

##### Max Unavailable

`.spec.strategy.rollingUpdate.maxUnavailable` is an optional field that specifies the
maximum number of pods that can be unavailable during the update process.
Value can be an absolute number (ex: 5) or a percentage of desired pods (ex:
10%).
Absolute number is calculated from percentage by rounding up.
This can not be 0 if `.spec.strategy.rollingUpdate.maxSurge` is 0.
By default, a fixed value of 1 is used.
Example: when this is set to 30%, the old RC can be scaled down to
70% of desired pods immediately when the rolling update starts. Once new pods are
ready, old RC can be scaled down further, followed by scaling up the new RC,
ensuring that the total number of pods available at all times during the
update is at least 70% of desired pods.

##### Max Surge

`.spec.strategy.rollingUpdate.maxSurge` is an optional field that specifies the
maximum number of pods that can be created above the desired number of pods.
Value can be an absolute number (ex: 5) or a percentage of desired pods (ex:
10%).
This can not be 0 if MaxUnavailable is 0.
Absolute number is calculated from percentage by rounding up.
By default, a value of 1 is used.
Example: when this is set to 30%, the new RC can be scaled up immediately when
the rolling update starts, such that the total number of old and new pods do not exceed
130% of desired pods. Once old pods have been killed,
new RC can be scaled up further, ensuring that total number of pods running
at any time during the update is atmost 130% of desired pods.

##### Min Ready Seconds

`.spec.strategy.rollingUpdate.minReadySeconds` is an optional field that specifies the
minimum number of seconds for which a newly created pod should be ready
without any of its container crashing, for it to be considered available.
Defaults to 0 (pod will be considered available as soon as it is ready).
Note: This is not implemented yet.

## Alternative to Deployments

### kubectl rolling update

[Kubectl rolling update](kubectl/kubectl_rolling-update.md) also updates pods and replication controllers in a similar fashion.
But deployments is declarative and is server side.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/deployments.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

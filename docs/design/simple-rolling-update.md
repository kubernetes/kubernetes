<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Simple rolling update

This is a lightweight design document for simple [rolling update](../user-guide/kubectl/kubectl_rolling-update.md) in `kubectl`.

Complete execution flow can be found [here](#execution-details). See the [example of rolling update](../user-guide/update-demo/) for more information.

### Lightweight rollout

Assume that we have a current replication controller named `foo` and it is running image `image:v1`

`kubectl rolling-update foo [foo-v2] --image=myimage:v2`

If the user doesn't specify a name for the 'next' replication controller, then the 'next' replication controller is renamed to
the name of the original replication controller.

Obviously there is a race here, where if you kill the client between delete foo, and creating the new version of 'foo' you might be surprised about what is there, but I think that's ok.
See [Recovery](#recovery) below

If the user does specify a name for the 'next' replication controller, then the 'next' replication controller is retained with its existing name,
and the old 'foo' replication controller is deleted.  For the purposes of the rollout, we add a unique-ifying label `kubernetes.io/deployment` to both the `foo` and `foo-next` replication controllers.
The value of that label is the hash of the complete JSON representation of the`foo-next` or`foo` replication controller.  The name of this label can be overridden by the user with the `--deployment-label-key` flag.

#### Recovery

If a rollout fails or is terminated in the middle, it is important that the user be able to resume the roll out.
To facilitate recovery in the case of a crash of the updating process itself, we add the following annotations to each replication controller in the `kubernetes.io/` annotation namespace:
   * `desired-replicas` The desired number of replicas for this replication controller (either N or zero)
   * `update-partner` A pointer to the replication controller resource that is the other half of this update (syntax `<name>` the namespace is assumed to be identical to the namespace of this replication controller.)

Recovery is achieved by issuing the same command again:

```sh
kubectl rolling-update foo [foo-v2] --image=myimage:v2
```

Whenever the rolling update command executes, the kubectl client looks for replication controllers called `foo` and `foo-next`, if they exist, an attempt is
made to roll `foo` to `foo-next`.  If `foo-next` does not exist, then it is created, and the rollout is a new rollout.  If `foo` doesn't exist, then
it is assumed that the rollout is nearly completed, and `foo-next` is renamed to `foo`.  Details of the execution flow are given below.


### Aborting a rollout

Abort is assumed to want to reverse a rollout in progress.

`kubectl rolling-update foo [foo-v2] --rollback`

This is really just semantic sugar for:

`kubectl rolling-update foo-v2 foo`

With the added detail that it moves the `desired-replicas` annotation from `foo-v2` to `foo`


### Execution Details

For the purposes of this example, assume that we are rolling from `foo` to `foo-next` where the only change is an image update from `v1` to `v2`

If the user doesn't specify a `foo-next` name, then it is either discovered from the `update-partner` annotation on `foo`.  If that annotation doesn't exist,
then `foo-next` is synthesized using the pattern `<controller-name>-<hash-of-next-controller-JSON>`

#### Initialization

   * If `foo` and `foo-next` do not exist:
      * Exit, and indicate an error to the user, that the specified controller doesn't exist.
   * If `foo` exists, but `foo-next` does not:
      * Create `foo-next` populate it with the `v2` image, set `desired-replicas` to `foo.Spec.Replicas`
      * Goto Rollout
   * If `foo-next` exists, but `foo` does not:
      * Assume that we are in the rename phase.
      * Goto Rename
   * If both `foo` and `foo-next` exist:
      * Assume that we are in a partial rollout
      * If `foo-next` is missing the `desired-replicas` annotation
         * Populate the `desired-replicas` annotation to `foo-next` using the current size of `foo`
      * Goto Rollout

#### Rollout

   * While size of `foo-next` < `desired-replicas` annotation on `foo-next`
      * increase size of `foo-next`
      * if size of `foo` > 0
         decrease size of `foo`
   * Goto Rename

#### Rename

   * delete `foo`
   * create `foo` that is identical to `foo-next`
   * delete `foo-next`

#### Abort

   * If `foo-next` doesn't exist
      * Exit and indicate to the user that they may want to simply do a new rollout with the old version
   * If `foo` doesn't exist
      * Exit and indicate not found to the user
   * Otherwise, `foo-next` and `foo` both exist
      * Set `desired-replicas` annotation on `foo` to match the annotation on `foo-next`
      * Goto Rollout with `foo` and `foo-next` trading places.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/simple-rolling-update.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

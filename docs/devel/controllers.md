<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
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

# Writing Controllers

A Kubernetes controller is simply an optimization on shell script that does repeated lists against the API server.

## Guidelines

When you’re writing controllers, there are few guidelines that will help make sure you get the results and performance
you’re looking for.

1. Operate on one item at a time.  If you use a `workqueue.Interface`, you’ll be able to queue changes for a
 particular resource and later pop them in multiple “worker” gofuncs with a guarantee that no two gofuncs will
 work on the same item at the same time.

 Many controllers must trigger off multiple resources (I need to "check X if Y changes"), but nearly all controllers
 can collapse those into a queue of “check this X” based on relationships.  For instance, a ReplicaSetController needs
 to react to a pod being deleted, but it does that by finding the related ReplicaSets and queuing those.


1. Random ordering between resources. When controllers queue off multiple types of resources, there is no guarantee
 of ordering amongst those resources.

 Distinct watches are updated independently.  Even with an objective ordering of “created resourceA/X” and “created
 resourceB/Y”, your controller could observe “created resourceB/Y” and “created resourceA/X”.


1. Level driven, not edge driven.  Just like having a shell script that isn’t running all the time, your controller
 may be off for an indeterminate amount of time before running again.

 If an API object appears with a marker value of `true`, you can’t count on having seen it turn from `false` to `true`,
 only that you now observe it being `true`.  Even an API watch suffers from this problem, so be sure that you’re not
 counting on seeing a change unless your controller is also marking the information it last made the decision on in
 the object's status.


1. Use `SharedInformers`.  `SharedInformers` provide hooks to receive notifications of adds, updates, and deletes for
 a particular resource.  They also provide convenience functions for accessing shared caches and determining when a
 cache is primed.

 Use the factory methods down in https://github.com/kubernetes/kubernetes/blob/master/pkg/controller/framework/informers/factory.go
 to ensure that you are sharing the same instance of the cache as everyone else.

 This saves us connections against the API server, duplicate serialization costs server-side, duplicate deserialization
 costs controller-side, and duplicate caching costs controller-side.

 You may see other mechanisms like reflectors and deltafifos driving controllers.  Those were older mechanisms that we
 later used to build the `SharedInformers`.  You should avoid using them in new controllers


1. Never mutate original objects!  Caches are shared across controllers, this means that if you mutate your "copy"
 (actually a reference or shallow copy) of an object, you’ll mess up other controllers (not just your own).

 The most common point of failure is making a shallow copy, then mutating a map, like `Annotations`.  Use
 `api.Scheme.Copy` to make a deep copy.


1. Wait for your secondary caches.  Many controllers have primary and secondary resources.  Primary resources are the
 resources that you’ll be updating `Status` for.  Secondary resources are resources that you’ll be managing
 (creating/deleting) or using for lookups.

 Use the `framework.WaitForCacheSync` function to wait for your secondary caches before starting your primary sync
 functions.  This will make sure that things like a Pod count for a ReplicaSet isn’t working off of known out of date
 information that results in thrashing.


1. Percolate errors to the top level for consistent re-queuing.  We have a  `workqueue.RateLimitingInterface` to allow
 simple requeuing with reasonable backoffs.

 Your main controller func should return an error when requeuing is necessary.  When it isn’t, it should use
 `utilruntime.HandleError` and return nil instead.  This makes it very easy for reviewers to inspect error handling
 cases and to be confident that your controller doesn’t accidentally lose things it should retry for.


1. Watches and Informers will “sync”.  Periodically, they will deliver every matching object in the cluster to your
 `Update` method.  This is good for cases where you may need to take additional action on the object, but sometimes you
 know there won’t be more work to do.

 In cases where you are *certain* that you don't need to requeue items when there are no new changes, you can compare the
 resource version of the old and new objects.  If they are the same, you skip requeuing the work.  Be careful when you
 do this.  If you ever skip requeuing your item on failures, you could fail, not requeue, and then never retry that
 item again.


## Rough Structure

Overall, your controller should look something like this:

```go
type Controller struct{
    podLister cache.StoreToPodLister
    queue workqueue.RateLimitingInterface
}

func (c *Controller) Run(threadiness int, stopCh chan struct{}){
    	if !framework.WaitForCacheSync(stopCh, c.podStoreSynced, c.nodeStoreSynced) {
		return
	}

	for i := 0; i < threadiness; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	<-stopCh
}

func (c *Controller) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *Controller) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncHandler(key.(string))
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", key, err))
	c.queue.AddRateLimited(key)

	return true
}

```


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/devel/controllers.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

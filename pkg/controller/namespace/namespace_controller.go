/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package namespace

import (
	"time"

	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/client/typed/dynamic"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const (
	// namespaceDeletionGracePeriod is the time period to wait before processing a received namespace event.
	// This allows time for the following to occur:
	// * lifecycle admission plugins on HA apiservers to also observe a namespace
	//   deletion and prevent new objects from being created in the terminating namespace
	// * non-leader etcd servers to observe last-minute object creations in a namespace
	//   so this controller's cleanup can actually clean up all objects
	namespaceDeletionGracePeriod = 5 * time.Second
)

// NamespaceController is responsible for performing actions dependent upon a namespace phase
type NamespaceController struct {
	// client that purges namespace content, must have list/delete privileges on all content
	kubeClient clientset.Interface
	// clientPool manages a pool of dynamic clients
	clientPool dynamic.ClientPool
	// store that holds the namespaces
	store cache.Store
	// controller that observes the namespaces
	controller *cache.Controller
	// namespaces that have been queued up for processing by workers
	queue workqueue.RateLimitingInterface
	// function to list of preferred resources for namespace deletion
	discoverResourcesFn func() ([]*metav1.APIResourceList, error)
	// opCache is a cache to remember if a particular operation is not supported to aid dynamic client.
	opCache *operationNotSupportedCache
	// finalizerToken is the finalizer token managed by this controller
	finalizerToken v1.FinalizerName
}

// NewNamespaceController creates a new NamespaceController
func NewNamespaceController(
	kubeClient clientset.Interface,
	clientPool dynamic.ClientPool,
	discoverResourcesFn func() ([]*metav1.APIResourceList, error),
	resyncPeriod time.Duration,
	finalizerToken v1.FinalizerName) *NamespaceController {

	opCache := &operationNotSupportedCache{
		m: make(map[operationKey]bool),
	}

	// pre-fill opCache with the discovery info
	//
	// TODO(sttts): get rid of opCache and http 405 logic around it and trust discovery info
	resources, err := discoverResourcesFn()
	if err != nil {
		glog.Fatalf("Failed to get supported resources: %v", err)
	}
	deletableGroupVersionResources := []schema.GroupVersionResource{}
	for _, rl := range resources {
		gv, err := schema.ParseGroupVersion(rl.GroupVersion)
		if err != nil {
			glog.Errorf("Failed to parse GroupVersion %q, skipping: %v", rl.GroupVersion, err)
			continue
		}

		for _, r := range rl.APIResources {
			gvr := schema.GroupVersionResource{Group: gv.Group, Version: gv.Version, Resource: r.Name}
			verbs := sets.NewString([]string(r.Verbs)...)

			if !verbs.Has("delete") {
				glog.V(6).Infof("Skipping resource %v because it cannot be deleted.", gvr)
			}

			for _, op := range []operation{operationList, operationDeleteCollection} {
				if !verbs.Has(string(op)) {
					opCache.setNotSupported(operationKey{op: op, gvr: gvr})
				}
			}

			deletableGroupVersionResources = append(deletableGroupVersionResources, gvr)
		}
	}

	// create the controller so we can inject the enqueue function
	namespaceController := &NamespaceController{
		kubeClient:          kubeClient,
		clientPool:          clientPool,
		queue:               workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "namespace"),
		discoverResourcesFn: discoverResourcesFn,
		opCache:             opCache,
		finalizerToken:      finalizerToken,
	}

	if kubeClient != nil && kubeClient.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("namespace_controller", kubeClient.Core().RESTClient().GetRateLimiter())
	}

	// configure the backing store/controller
	store, controller := cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
				return kubeClient.Core().Namespaces().List(options)
			},
			WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
				return kubeClient.Core().Namespaces().Watch(options)
			},
		},
		&v1.Namespace{},
		resyncPeriod,
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				namespace := obj.(*v1.Namespace)
				namespaceController.enqueueNamespace(namespace)
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				namespace := newObj.(*v1.Namespace)
				namespaceController.enqueueNamespace(namespace)
			},
		},
	)

	namespaceController.store = store
	namespaceController.controller = controller
	return namespaceController
}

// enqueueNamespace adds an object to the controller work queue
// obj could be an *v1.Namespace, or a DeletionFinalStateUnknown item.
func (nm *NamespaceController) enqueueNamespace(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	// delay processing namespace events to allow HA api servers to observe namespace deletion,
	// and HA etcd servers to observe last minute object creations inside the namespace
	nm.queue.AddAfter(key, namespaceDeletionGracePeriod)
}

// worker processes the queue of namespace objects.
// Each namespace can be in the queue at most once.
// The system ensures that no two workers can process
// the same namespace at the same time.
func (nm *NamespaceController) worker() {
	workFunc := func() bool {
		key, quit := nm.queue.Get()
		if quit {
			return true
		}
		defer nm.queue.Done(key)

		err := nm.syncNamespaceFromKey(key.(string))
		if err == nil {
			// no error, forget this entry and return
			nm.queue.Forget(key)
			return false
		}

		if estimate, ok := err.(*contentRemainingError); ok {
			t := estimate.Estimate/2 + 1
			glog.V(4).Infof("Content remaining in namespace %s, waiting %d seconds", key, t)
			nm.queue.AddAfter(key, time.Duration(t)*time.Second)
		} else {
			// rather than wait for a full resync, re-add the namespace to the queue to be processed
			nm.queue.AddRateLimited(key)
			utilruntime.HandleError(err)
		}
		return false
	}

	for {
		quit := workFunc()

		if quit {
			return
		}
	}
}

// syncNamespaceFromKey looks for a namespace with the specified key in its store and synchronizes it
func (nm *NamespaceController) syncNamespaceFromKey(key string) (err error) {
	startTime := time.Now()
	defer glog.V(4).Infof("Finished syncing namespace %q (%v)", key, time.Now().Sub(startTime))

	obj, exists, err := nm.store.GetByKey(key)
	if !exists {
		glog.Infof("Namespace has been deleted %v", key)
		return nil
	}
	if err != nil {
		glog.Errorf("Unable to retrieve namespace %v from store: %v", key, err)
		nm.queue.Add(key)
		return err
	}
	namespace := obj.(*v1.Namespace)
	return syncNamespace(nm.kubeClient, nm.clientPool, nm.opCache, nm.discoverResourcesFn, namespace, nm.finalizerToken)
}

// Run starts observing the system with the specified number of workers.
func (nm *NamespaceController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	go nm.controller.Run(stopCh)
	for i := 0; i < workers; i++ {
		go wait.Until(nm.worker, time.Second, stopCh)
	}
	<-stopCh
	glog.Infof("Shutting down NamespaceController")
	nm.queue.ShutDown()
}

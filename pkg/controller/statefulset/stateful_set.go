/*
Copyright 2016 The Kubernetes Authors.

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

package statefulset

import (
	"fmt"
	"reflect"
	"sort"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	appsinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/apps/v1beta1"
	coreinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/core/v1"
	appslisters "k8s.io/kubernetes/pkg/client/listers/apps/v1beta1"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

const (
	// period to relist statefulsets and verify pets
	statefulSetResyncPeriod = 30 * time.Second
)

// StatefulSetController controls statefulsets.
type StatefulSetController struct {
	// client interface
	kubeClient clientset.Interface
	// control returns an interface capable of syncing a stateful set.
	// Abstracted out for testing.
	control StatefulSetControlInterface
	// podLister is able to list/get pods from a shared informer's store
	podLister corelisters.PodLister
	// podListerSynced returns true if the pod shared informer has synced at least once
	podListerSynced cache.InformerSynced
	// setLister is able to list/get stateful sets from a shared informer's store
	setLister appslisters.StatefulSetLister
	// setListerSynced returns true if the stateful set shared informer has synced at least once
	setListerSynced cache.InformerSynced
	// pvcListerSynced returns true if the pvc shared informer has synced at least once
	pvcListerSynced cache.InformerSynced
	// StatefulSets that need to be synced.
	queue workqueue.RateLimitingInterface
}

// NewStatefulSetController creates a new statefulset controller.
func NewStatefulSetController(
	podInformer coreinformers.PodInformer,
	setInformer appsinformers.StatefulSetInformer,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	kubeClient clientset.Interface,
) *StatefulSetController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(kubeClient.Core().RESTClient()).Events("")})
	recorder := eventBroadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "statefulset"})

	ssc := &StatefulSetController{
		kubeClient: kubeClient,
		control: NewDefaultStatefulSetControl(
			NewRealStatefulPodControl(
				kubeClient,
				setInformer.Lister(),
				podInformer.Lister(),
				pvcInformer.Lister(),
				recorder,
			),
		),
		pvcListerSynced: pvcInformer.Informer().HasSynced,
		queue:           workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "statefulset"),
	}

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		// lookup the statefulset and enqueue
		AddFunc: ssc.addPod,
		// lookup current and old statefulset if labels changed
		UpdateFunc: ssc.updatePod,
		// lookup statefulset accounting for deletion tombstones
		DeleteFunc: ssc.deletePod,
	})
	ssc.podLister = podInformer.Lister()
	ssc.podListerSynced = podInformer.Informer().HasSynced

	setInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc: ssc.enqueueStatefulSet,
			UpdateFunc: func(old, cur interface{}) {
				oldPS := old.(*apps.StatefulSet)
				curPS := cur.(*apps.StatefulSet)
				if oldPS.Status.Replicas != curPS.Status.Replicas {
					glog.V(4).Infof("Observed updated replica count for StatefulSet: %v, %d->%d", curPS.Name, oldPS.Status.Replicas, curPS.Status.Replicas)
				}
				ssc.enqueueStatefulSet(cur)
			},
			DeleteFunc: ssc.enqueueStatefulSet,
		},
		statefulSetResyncPeriod,
	)
	ssc.setLister = setInformer.Lister()
	ssc.setListerSynced = setInformer.Informer().HasSynced

	// TODO: Watch volumes
	return ssc
}

// Run runs the statefulset controller.
func (ssc *StatefulSetController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer ssc.queue.ShutDown()

	glog.Infof("Starting statefulset controller")

	if !cache.WaitForCacheSync(stopCh, ssc.podListerSynced, ssc.setListerSynced, ssc.pvcListerSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(ssc.worker, time.Second, stopCh)
	}

	<-stopCh
	glog.Infof("Shutting down statefulset controller")
}

// addPod adds the statefulset for the pod to the sync queue
func (ssc *StatefulSetController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	glog.V(4).Infof("Pod %s created, labels: %+v", pod.Name, pod.Labels)
	set := ssc.getStatefulSetForPod(pod)
	if set == nil {
		return
	}
	ssc.enqueueStatefulSet(set)
}

// updatePod adds the statefulset for the current and old pods to the sync queue.
// If the labels of the pod didn't change, this method enqueues a single statefulset.
func (ssc *StatefulSetController) updatePod(old, cur interface{}) {
	curPod := cur.(*v1.Pod)
	oldPod := old.(*v1.Pod)
	if curPod.ResourceVersion == oldPod.ResourceVersion {
		// Periodic resync will send update events for all known pods.
		// Two different versions of the same pod will always have different RVs.
		return
	}
	set := ssc.getStatefulSetForPod(curPod)
	if set == nil {
		return
	}
	ssc.enqueueStatefulSet(set)
	// TODO will we need this going forward with controller ref impl?
	if !reflect.DeepEqual(curPod.Labels, oldPod.Labels) {
		if oldSet := ssc.getStatefulSetForPod(oldPod); oldSet != nil {
			ssc.enqueueStatefulSet(oldSet)
		}
	}
}

// deletePod enqueues the statefulset for the pod accounting for deletion tombstones.
func (ssc *StatefulSetController) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)

	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new StatefulSet will not be woken up till the periodic resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("couldn't get object from tombstone %+v", obj))
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("tombstone contained object that is not a pod %+v", obj))
			return
		}
	}
	glog.V(4).Infof("Pod %s/%s deleted through %v.", pod.Namespace, pod.Name, utilruntime.GetCaller())
	if set := ssc.getStatefulSetForPod(pod); set != nil {
		ssc.enqueueStatefulSet(set)
	}
}

// getPodsForStatefulSets returns the pods that match the selectors of the given statefulset.
func (ssc *StatefulSetController) getPodsForStatefulSet(set *apps.StatefulSet) ([]*v1.Pod, error) {
	sel, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return []*v1.Pod{}, err
	}
	return ssc.podLister.Pods(set.Namespace).List(sel)
}

// getStatefulSetForPod returns the StatefulSet managing the given pod.
func (ssc *StatefulSetController) getStatefulSetForPod(pod *v1.Pod) *apps.StatefulSet {
	sets, err := ssc.setLister.GetPodStatefulSets(pod)
	if err != nil {
		glog.V(4).Infof("No StatefulSets found for pod %v, StatefulSet controller will avoid syncing", pod.Name)
		return nil
	}
	// More than one set is selecting the same Pod
	if len(sets) > 1 {
		utilruntime.HandleError(
			fmt.Errorf(
				"user error: more than one StatefulSet is selecting pods with labels: %+v",
				pod.Labels))
		// The timestamp sort should not be necessary because we will enforce the CreatedBy requirement by
		// name
		sort.Sort(overlappingStatefulSets(sets))
		// return the first created set for which pod is a member
		for i := range sets {
			if isMemberOf(sets[i], pod) {
				return sets[i]
			}
		}
		glog.V(4).Infof("No StatefulSets found for pod %v, StatefulSet controller will avoid syncing", pod.Name)
		return nil
	}
	return sets[0]

}

// enqueueStatefulSet enqueues the given statefulset in the work queue.
func (ssc *StatefulSetController) enqueueStatefulSet(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Cound't get key for object %+v: %v", obj, err))
		return
	}
	ssc.queue.Add(key)
}

// processNextWorkItem dequeues items, processes them, and marks them done. It enforces that the syncHandler is never
// invoked concurrently with the same key.
func (ssc *StatefulSetController) processNextWorkItem() bool {
	key, quit := ssc.queue.Get()
	if quit {
		return false
	}
	defer ssc.queue.Done(key)
	if err := ssc.sync(key.(string)); err != nil {
		utilruntime.HandleError(fmt.Errorf("Error syncing StatefulSet %v, requeuing: %v", key.(string), err))
		ssc.queue.AddRateLimited(key)
	} else {
		ssc.queue.Forget(key)
	}
	return true
}

// worker runs a worker goroutine that invokes processNextWorkItem until the the controller's queue is closed
func (ssc *StatefulSetController) worker() {
	for ssc.processNextWorkItem() {
	}
}

// sync syncs the given statefulset.
func (ssc *StatefulSetController) sync(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing statefulset %q (%v)", key, time.Now().Sub(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	set, err := ssc.setLister.StatefulSets(namespace).Get(name)
	if errors.IsNotFound(err) {
		glog.Infof("StatefulSet has been deleted %v", key)
		return nil
	}
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Unable to retrieve StatefulSet %v from store: %v", key, err))
		return err
	}

	pods, err := ssc.getPodsForStatefulSet(set)
	if err != nil {
		return err
	}

	return ssc.syncStatefulSet(set, pods)
}

// syncStatefulSet syncs a tuple of (statefulset, []*v1.Pod).
func (ssc *StatefulSetController) syncStatefulSet(set *apps.StatefulSet, pods []*v1.Pod) error {
	glog.V(2).Infof("Syncing StatefulSet %v/%v with %d pods", set.Namespace, set.Name, len(pods))
	if err := ssc.control.UpdateStatefulSet(set, pods); err != nil {
		glog.V(2).Infof("Error syncing StatefulSet %s/%s with %d pods : %s", set.Namespace, set.Name, err)
		return err
	}
	glog.V(2).Infof("Succesfully synced StatefulSet %s/%s successful", set.Namespace, set.Name)
	return nil
}

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

package disruption

import (
	"fmt"
	"reflect"
	"time"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	policy "k8s.io/kubernetes/pkg/apis/policy/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	policyclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/policy/v1beta1"
	appsinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/apps/v1beta1"
	coreinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/core/v1"
	extensionsinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/extensions/v1beta1"
	policyinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/policy/v1beta1"
	appslisters "k8s.io/kubernetes/pkg/client/listers/apps/v1beta1"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	extensionslisters "k8s.io/kubernetes/pkg/client/listers/extensions/v1beta1"
	policylisters "k8s.io/kubernetes/pkg/client/listers/policy/v1beta1"
	"k8s.io/kubernetes/pkg/controller"

	"github.com/golang/glog"
)

const statusUpdateRetries = 2

// DeletionTimeout sets maximum time from the moment a pod is added to DisruptedPods in PDB.Status
// to the time when the pod is expected to be seen by PDB controller as having been marked for deletion.
// If the pod was not marked for deletion during that time it is assumed that it won't be deleted at
// all and the corresponding entry can be removed from pdb.Status.DisruptedPods. It is assumed that
// pod/pdb apiserver to controller latency is relatively small (like 1-2sec) so the below value should
// be more than enough.
// If the controller is running on a different node it is important that the two nodes have synced
// clock (via ntp for example). Otherwise PodDisruptionBudget controller may not provide enough
// protection against unwanted pod disruptions.
const DeletionTimeout = 2 * 60 * time.Second

type updater func(*policy.PodDisruptionBudget) error

type DisruptionController struct {
	kubeClient clientset.Interface

	pdbLister       policylisters.PodDisruptionBudgetLister
	pdbListerSynced cache.InformerSynced

	podLister       corelisters.PodLister
	podListerSynced cache.InformerSynced

	rcLister       corelisters.ReplicationControllerLister
	rcListerSynced cache.InformerSynced

	rsLister       extensionslisters.ReplicaSetLister
	rsListerSynced cache.InformerSynced

	dLister       extensionslisters.DeploymentLister
	dListerSynced cache.InformerSynced

	ssLister       appslisters.StatefulSetLister
	ssListerSynced cache.InformerSynced

	// PodDisruptionBudget keys that need to be synced.
	queue        workqueue.RateLimitingInterface
	recheckQueue workqueue.DelayingInterface

	broadcaster record.EventBroadcaster
	recorder    record.EventRecorder

	getUpdater func() updater
}

// controllerAndScale is used to return (controller, scale) pairs from the
// controller finder functions.
type controllerAndScale struct {
	types.UID
	scale int32
}

// podControllerFinder is a function type that maps a pod to a list of
// controllers and their scale.
type podControllerFinder func(*v1.Pod) ([]controllerAndScale, error)

func NewDisruptionController(
	podInformer coreinformers.PodInformer,
	pdbInformer policyinformers.PodDisruptionBudgetInformer,
	rcInformer coreinformers.ReplicationControllerInformer,
	rsInformer extensionsinformers.ReplicaSetInformer,
	dInformer extensionsinformers.DeploymentInformer,
	ssInformer appsinformers.StatefulSetInformer,
	kubeClient clientset.Interface,
) *DisruptionController {
	dc := &DisruptionController{
		kubeClient:   kubeClient,
		queue:        workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "disruption"),
		recheckQueue: workqueue.NewNamedDelayingQueue("disruption-recheck"),
		broadcaster:  record.NewBroadcaster(),
	}
	dc.recorder = dc.broadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "controllermanager"})

	dc.getUpdater = func() updater { return dc.writePdbStatus }

	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    dc.addPod,
		UpdateFunc: dc.updatePod,
		DeleteFunc: dc.deletePod,
	})
	dc.podLister = podInformer.Lister()
	dc.podListerSynced = podInformer.Informer().HasSynced

	pdbInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    dc.addDb,
			UpdateFunc: dc.updateDb,
			DeleteFunc: dc.removeDb,
		},
		30*time.Second,
	)
	dc.pdbLister = pdbInformer.Lister()
	dc.pdbListerSynced = pdbInformer.Informer().HasSynced

	dc.rcLister = rcInformer.Lister()
	dc.rcListerSynced = rcInformer.Informer().HasSynced

	dc.rsLister = rsInformer.Lister()
	dc.rsListerSynced = rsInformer.Informer().HasSynced

	dc.dLister = dInformer.Lister()
	dc.dListerSynced = dInformer.Informer().HasSynced

	dc.ssLister = ssInformer.Lister()
	dc.ssListerSynced = ssInformer.Informer().HasSynced

	return dc
}

// TODO(mml): When controllerRef is implemented (#2210), we *could* simply
// return controllers without their scales, and access scale type-generically
// via the scale subresource.  That may not be as much of a win as it sounds,
// however.  We are accessing everything through the pkg/client/cache API that
// we have to set up and tune to the types we know we'll be accessing anyway,
// and we may well need further tweaks just to be able to access scale
// subresources.
func (dc *DisruptionController) finders() []podControllerFinder {
	return []podControllerFinder{dc.getPodReplicationControllers, dc.getPodDeployments, dc.getPodReplicaSets,
		dc.getPodStatefulSets}
}

// getPodReplicaSets finds replicasets which have no matching deployments.
func (dc *DisruptionController) getPodReplicaSets(pod *v1.Pod) ([]controllerAndScale, error) {
	cas := []controllerAndScale{}
	rss, err := dc.rsLister.GetPodReplicaSets(pod)
	// GetPodReplicaSets returns an error only if no ReplicaSets are found.  We
	// don't return that as an error to the caller.
	if err != nil {
		return cas, nil
	}
	controllerScale := map[types.UID]int32{}
	for _, rs := range rss {
		// GetDeploymentsForReplicaSet returns an error only if no matching
		// deployments are found.
		_, err := dc.dLister.GetDeploymentsForReplicaSet(rs)
		if err == nil { // A deployment was found, so this finder will not count this RS.
			continue
		}
		controllerScale[rs.UID] = *(rs.Spec.Replicas)
	}

	for uid, scale := range controllerScale {
		cas = append(cas, controllerAndScale{UID: uid, scale: scale})
	}

	return cas, nil
}

// getPodStatefulSet returns the statefulset managing the given pod.
func (dc *DisruptionController) getPodStatefulSets(pod *v1.Pod) ([]controllerAndScale, error) {
	cas := []controllerAndScale{}
	ss, err := dc.ssLister.GetPodStatefulSets(pod)

	// GetPodStatefulSets returns an error only if no StatefulSets are found. We
	// don't return that as an error to the caller.
	if err != nil {
		return cas, nil
	}

	controllerScale := map[types.UID]int32{}
	for _, s := range ss {
		controllerScale[s.UID] = *(s.Spec.Replicas)
	}

	for uid, scale := range controllerScale {
		cas = append(cas, controllerAndScale{UID: uid, scale: scale})
	}

	return cas, nil
}

// getPodDeployments finds deployments for any replicasets which are being managed by deployments.
func (dc *DisruptionController) getPodDeployments(pod *v1.Pod) ([]controllerAndScale, error) {
	cas := []controllerAndScale{}
	rss, err := dc.rsLister.GetPodReplicaSets(pod)
	// GetPodReplicaSets returns an error only if no ReplicaSets are found.  We
	// don't return that as an error to the caller.
	if err != nil {
		return cas, nil
	}
	controllerScale := map[types.UID]int32{}
	for _, rs := range rss {
		ds, err := dc.dLister.GetDeploymentsForReplicaSet(rs)
		// GetDeploymentsForReplicaSet returns an error only if no matching
		// deployments are found.  In that case we skip this ReplicaSet.
		if err != nil {
			continue
		}
		for _, d := range ds {
			controllerScale[d.UID] = *(d.Spec.Replicas)
		}
	}

	for uid, scale := range controllerScale {
		cas = append(cas, controllerAndScale{UID: uid, scale: scale})
	}

	return cas, nil
}

func (dc *DisruptionController) getPodReplicationControllers(pod *v1.Pod) ([]controllerAndScale, error) {
	cas := []controllerAndScale{}
	rcs, err := dc.rcLister.GetPodControllers(pod)
	if err == nil {
		for _, rc := range rcs {
			cas = append(cas, controllerAndScale{UID: rc.UID, scale: *(rc.Spec.Replicas)})
		}
	}
	return cas, nil
}

func (dc *DisruptionController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer dc.queue.ShutDown()

	glog.V(0).Infof("Starting disruption controller")

	if !cache.WaitForCacheSync(stopCh, dc.podListerSynced, dc.pdbListerSynced, dc.rcListerSynced, dc.rsListerSynced, dc.dListerSynced, dc.ssListerSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	if dc.kubeClient != nil {
		glog.V(0).Infof("Sending events to api server.")
		dc.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: v1core.New(dc.kubeClient.Core().RESTClient()).Events("")})
	} else {
		glog.V(0).Infof("No api server defined - no events will be sent to API server.")
	}
	go wait.Until(dc.worker, time.Second, stopCh)
	go wait.Until(dc.recheckWorker, time.Second, stopCh)

	<-stopCh
	glog.V(0).Infof("Shutting down disruption controller")
}

func (dc *DisruptionController) addDb(obj interface{}) {
	pdb := obj.(*policy.PodDisruptionBudget)
	glog.V(4).Infof("add DB %q", pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) updateDb(old, cur interface{}) {
	// TODO(mml) ignore updates where 'old' is equivalent to 'cur'.
	pdb := cur.(*policy.PodDisruptionBudget)
	glog.V(4).Infof("update DB %q", pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) removeDb(obj interface{}) {
	pdb := obj.(*policy.PodDisruptionBudget)
	glog.V(4).Infof("remove DB %q", pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	glog.V(4).Infof("addPod called on pod %q", pod.Name)
	pdb := dc.getPdbForPod(pod)
	if pdb == nil {
		glog.V(4).Infof("No matching pdb for pod %q", pod.Name)
		return
	}
	glog.V(4).Infof("addPod %q -> PDB %q", pod.Name, pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) updatePod(old, cur interface{}) {
	pod := cur.(*v1.Pod)
	glog.V(4).Infof("updatePod called on pod %q", pod.Name)
	pdb := dc.getPdbForPod(pod)
	if pdb == nil {
		glog.V(4).Infof("No matching pdb for pod %q", pod.Name)
		return
	}
	glog.V(4).Infof("updatePod %q -> PDB %q", pod.Name, pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) deletePod(obj interface{}) {
	pod, ok := obj.(*v1.Pod)
	// When a delete is dropped, the relist will notice a pod in the store not
	// in the list, leading to the insertion of a tombstone object which contains
	// the deleted key/value. Note that this value might be stale. If the pod
	// changed labels the new ReplicaSet will not be woken up till the periodic
	// resync.
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			glog.Errorf("Couldn't get object from tombstone %+v", obj)
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			glog.Errorf("Tombstone contained object that is not a pod %+v", obj)
			return
		}
	}
	glog.V(4).Infof("deletePod called on pod %q", pod.Name)
	pdb := dc.getPdbForPod(pod)
	if pdb == nil {
		glog.V(4).Infof("No matching pdb for pod %q", pod.Name)
		return
	}
	glog.V(4).Infof("deletePod %q -> PDB %q", pod.Name, pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) enqueuePdb(pdb *policy.PodDisruptionBudget) {
	key, err := controller.KeyFunc(pdb)
	if err != nil {
		glog.Errorf("Cound't get key for PodDisruptionBudget object %+v: %v", pdb, err)
		return
	}
	dc.queue.Add(key)
}

func (dc *DisruptionController) enqueuePdbForRecheck(pdb *policy.PodDisruptionBudget, delay time.Duration) {
	key, err := controller.KeyFunc(pdb)
	if err != nil {
		glog.Errorf("Cound't get key for PodDisruptionBudget object %+v: %v", pdb, err)
		return
	}
	dc.recheckQueue.AddAfter(key, delay)
}

func (dc *DisruptionController) getPdbForPod(pod *v1.Pod) *policy.PodDisruptionBudget {
	// GetPodPodDisruptionBudgets returns an error only if no
	// PodDisruptionBudgets are found.  We don't return that as an error to the
	// caller.
	pdbs, err := dc.pdbLister.GetPodPodDisruptionBudgets(pod)
	if err != nil {
		glog.V(4).Infof("No PodDisruptionBudgets found for pod %v, PodDisruptionBudget controller will avoid syncing.", pod.Name)
		return nil
	}

	if len(pdbs) > 1 {
		msg := fmt.Sprintf("Pod %q/%q matches multiple PodDisruptionBudgets.  Chose %q arbitrarily.", pod.Namespace, pod.Name, pdbs[0].Name)
		glog.Warning(msg)
		dc.recorder.Event(pod, v1.EventTypeWarning, "MultiplePodDisruptionBudgets", msg)
	}
	return pdbs[0]
}

func (dc *DisruptionController) getPodsForPdb(pdb *policy.PodDisruptionBudget) ([]*v1.Pod, error) {
	sel, err := metav1.LabelSelectorAsSelector(pdb.Spec.Selector)
	if sel.Empty() {
		return []*v1.Pod{}, nil
	}
	if err != nil {
		return []*v1.Pod{}, err
	}
	pods, err := dc.podLister.Pods(pdb.Namespace).List(sel)
	if err != nil {
		return []*v1.Pod{}, err
	}
	// TODO: Do we need to copy here?
	result := make([]*v1.Pod, 0, len(pods))
	for i := range pods {
		result = append(result, &(*pods[i]))
	}
	return result, nil
}

func (dc *DisruptionController) worker() {
	for dc.processNextWorkItem() {
	}
}

func (dc *DisruptionController) processNextWorkItem() bool {
	dKey, quit := dc.queue.Get()
	if quit {
		return false
	}
	defer dc.queue.Done(dKey)

	err := dc.sync(dKey.(string))
	if err == nil {
		dc.queue.Forget(dKey)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("Error syncing PodDisruptionBudget %v, requeuing: %v", dKey.(string), err))
	dc.queue.AddRateLimited(dKey)

	return true
}

func (dc *DisruptionController) recheckWorker() {
	for dc.processNextRecheckWorkItem() {
	}
}

func (dc *DisruptionController) processNextRecheckWorkItem() bool {
	dKey, quit := dc.recheckQueue.Get()
	if quit {
		return false
	}
	defer dc.recheckQueue.Done(dKey)
	dc.queue.AddRateLimited(dKey)
	return true
}

func (dc *DisruptionController) sync(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing PodDisruptionBudget %q (%v)", key, time.Now().Sub(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	pdb, err := dc.pdbLister.PodDisruptionBudgets(namespace).Get(name)
	if errors.IsNotFound(err) {
		glog.V(4).Infof("PodDisruptionBudget %q has been deleted", key)
		return nil
	}
	if err != nil {
		return err
	}

	if err := dc.trySync(pdb); err != nil {
		glog.Errorf("Failed to sync pdb %s/%s: %v", pdb.Namespace, pdb.Name, err)
		return dc.failSafe(pdb)
	}

	return nil
}

func (dc *DisruptionController) trySync(pdb *policy.PodDisruptionBudget) error {
	pods, err := dc.getPodsForPdb(pdb)
	if err != nil {
		dc.recorder.Eventf(pdb, v1.EventTypeWarning, "NoPods", "Failed to get pods: %v", err)
		return err
	}
	if len(pods) == 0 {
		dc.recorder.Eventf(pdb, v1.EventTypeNormal, "NoPods", "No matching pods found")
	}

	expectedCount, desiredHealthy, err := dc.getExpectedPodCount(pdb, pods)
	if err != nil {
		dc.recorder.Eventf(pdb, v1.EventTypeNormal, "ExpectedPods", "Failed to calculate the number of expected pods: %v", err)
		return err
	}

	currentTime := time.Now()
	disruptedPods, recheckTime := dc.buildDisruptedPodMap(pods, pdb, currentTime)
	currentHealthy := countHealthyPods(pods, disruptedPods, currentTime)
	err = dc.updatePdbStatus(pdb, currentHealthy, desiredHealthy, expectedCount, disruptedPods)

	if err == nil && recheckTime != nil {
		// There is always at most one PDB waiting with a particular name in the queue,
		// and each PDB in the queue is associated with the lowest timestamp
		// that was supplied when a PDB with that name was added.
		dc.enqueuePdbForRecheck(pdb, recheckTime.Sub(currentTime))
	}
	return err
}

func (dc *DisruptionController) getExpectedPodCount(pdb *policy.PodDisruptionBudget, pods []*v1.Pod) (expectedCount, desiredHealthy int32, err error) {
	err = nil
	// TODO(davidopp): consider making the way expectedCount and rules about
	// permitted controller configurations (specifically, considering it an error
	// if a pod covered by a PDB has 0 controllers or > 1 controller) should be
	// handled the same way for integer and percentage minAvailable
	if pdb.Spec.MinAvailable.Type == intstr.Int {
		desiredHealthy = pdb.Spec.MinAvailable.IntVal
		expectedCount = int32(len(pods))
	} else if pdb.Spec.MinAvailable.Type == intstr.String {
		// When the user specifies a fraction of pods that must be available, we
		// use as the fraction's denominator
		// SUM_{all c in C} scale(c)
		// where C is the union of C_p1, C_p2, ..., C_pN
		// and each C_pi is the set of controllers controlling the pod pi

		// k8s only defines what will happens when 0 or 1 controllers control a
		// given pod.  We explicitly exclude the 0 controllers case here, and we
		// report an error if we find a pod with more than 1 controller.  Thus in
		// practice each C_pi is a set of exactly 1 controller.

		// A mapping from controllers to their scale.
		controllerScale := map[types.UID]int32{}

		// 1. Find the controller(s) for each pod.  If any pod has 0 controllers,
		// that's an error.  If any pod has more than 1 controller, that's also an
		// error.
		for _, pod := range pods {
			controllerCount := 0
			for _, finder := range dc.finders() {
				var controllers []controllerAndScale
				controllers, err = finder(pod)
				if err != nil {
					return
				}
				for _, controller := range controllers {
					controllerScale[controller.UID] = controller.scale
					controllerCount++
				}
			}
			if controllerCount == 0 {
				err = fmt.Errorf("asked for percentage, but found no controllers for pod %q", pod.Name)
				dc.recorder.Event(pdb, v1.EventTypeWarning, "NoControllers", err.Error())
				return
			} else if controllerCount > 1 {
				err = fmt.Errorf("pod %q has %v>1 controllers", pod.Name, controllerCount)
				dc.recorder.Event(pdb, v1.EventTypeWarning, "TooManyControllers", err.Error())
				return
			}
		}

		// 2. Add up all the controllers.
		expectedCount = 0
		for _, count := range controllerScale {
			expectedCount += count
		}

		// 3. Do the math.
		var dh int
		dh, err = intstr.GetValueFromIntOrPercent(&pdb.Spec.MinAvailable, int(expectedCount), true)
		if err != nil {
			return
		}
		desiredHealthy = int32(dh)
	}

	return
}

func countHealthyPods(pods []*v1.Pod, disruptedPods map[string]metav1.Time, currentTime time.Time) (currentHealthy int32) {
Pod:
	for _, pod := range pods {
		// Pod is beeing deleted.
		if pod.DeletionTimestamp != nil {
			continue
		}
		// Pod is expected to be deleted soon.
		if disruptionTime, found := disruptedPods[pod.Name]; found && disruptionTime.Time.Add(DeletionTimeout).After(currentTime) {
			continue
		}
		if v1.IsPodReady(pod) {
			currentHealthy++
			continue Pod
		}
	}

	return
}

// Builds new PodDisruption map, possibly removing items that refer to non-existing, already deleted
// or not-deleted at all items. Also returns an information when this check should be repeated.
func (dc *DisruptionController) buildDisruptedPodMap(pods []*v1.Pod, pdb *policy.PodDisruptionBudget, currentTime time.Time) (map[string]metav1.Time, *time.Time) {
	disruptedPods := pdb.Status.DisruptedPods
	result := make(map[string]metav1.Time)
	var recheckTime *time.Time

	if disruptedPods == nil || len(disruptedPods) == 0 {
		return result, recheckTime
	}
	for _, pod := range pods {
		if pod.DeletionTimestamp != nil {
			// Already being deleted.
			continue
		}
		disruptionTime, found := disruptedPods[pod.Name]
		if !found {
			// Pod not on the list.
			continue
		}
		expectedDeletion := disruptionTime.Time.Add(DeletionTimeout)
		if expectedDeletion.Before(currentTime) {
			glog.V(1).Infof("Pod %s/%s was expected to be deleted at %s but it wasn't, updating pdb %s/%s",
				pod.Namespace, pod.Name, disruptionTime.String(), pdb.Namespace, pdb.Name)
			dc.recorder.Eventf(pod, v1.EventTypeWarning, "NotDeleted", "Pod was expected by PDB %s/%s to be deleted but it wasn't",
				pdb.Namespace, pdb.Namespace)
		} else {
			if recheckTime == nil || expectedDeletion.Before(*recheckTime) {
				recheckTime = &expectedDeletion
			}
			result[pod.Name] = disruptionTime
		}
	}
	return result, recheckTime
}

// failSafe is an attempt to at least update the PodDisruptionsAllowed field to
// 0 if everything else has failed.  This is one place we
// implement the  "fail open" part of the design since if we manage to update
// this field correctly, we will prevent the /evict handler from approving an
// eviction when it may be unsafe to do so.
func (dc *DisruptionController) failSafe(pdb *policy.PodDisruptionBudget) error {
	obj, err := api.Scheme.DeepCopy(pdb)
	if err != nil {
		return err
	}
	newPdb := obj.(*policy.PodDisruptionBudget)
	newPdb.Status.PodDisruptionsAllowed = 0

	return dc.getUpdater()(newPdb)
}

func (dc *DisruptionController) updatePdbStatus(pdb *policy.PodDisruptionBudget, currentHealthy, desiredHealthy, expectedCount int32,
	disruptedPods map[string]metav1.Time) error {

	// We require expectedCount to be > 0 so that PDBs which currently match no
	// pods are in a safe state when their first pods appear but this controller
	// has not updated their status yet.  This isn't the only race, but it's a
	// common one that's easy to detect.
	disruptionsAllowed := currentHealthy - desiredHealthy
	if expectedCount <= 0 || disruptionsAllowed <= 0 {
		disruptionsAllowed = 0
	}

	if pdb.Status.CurrentHealthy == currentHealthy &&
		pdb.Status.DesiredHealthy == desiredHealthy &&
		pdb.Status.ExpectedPods == expectedCount &&
		pdb.Status.PodDisruptionsAllowed == disruptionsAllowed &&
		reflect.DeepEqual(pdb.Status.DisruptedPods, disruptedPods) &&
		pdb.Status.ObservedGeneration == pdb.Generation {
		return nil
	}

	obj, err := api.Scheme.DeepCopy(pdb)
	if err != nil {
		return err
	}
	newPdb := obj.(*policy.PodDisruptionBudget)

	newPdb.Status = policy.PodDisruptionBudgetStatus{
		CurrentHealthy:        currentHealthy,
		DesiredHealthy:        desiredHealthy,
		ExpectedPods:          expectedCount,
		PodDisruptionsAllowed: disruptionsAllowed,
		DisruptedPods:         disruptedPods,
		ObservedGeneration:    pdb.Generation,
	}

	return dc.getUpdater()(newPdb)
}

// refresh tries to re-GET the given PDB.  If there are any errors, it just
// returns the old PDB.  Intended to be used in a retry loop where it runs a
// bounded number of times.
func refresh(pdbClient policyclientset.PodDisruptionBudgetInterface, pdb *policy.PodDisruptionBudget) *policy.PodDisruptionBudget {
	newPdb, err := pdbClient.Get(pdb.Name, metav1.GetOptions{})
	if err == nil {
		return newPdb
	} else {
		return pdb
	}
}

func (dc *DisruptionController) writePdbStatus(pdb *policy.PodDisruptionBudget) error {
	pdbClient := dc.kubeClient.Policy().PodDisruptionBudgets(pdb.Namespace)
	st := pdb.Status

	var err error
	for i, pdb := 0, pdb; i < statusUpdateRetries; i, pdb = i+1, refresh(pdbClient, pdb) {
		pdb.Status = st
		if _, err = pdbClient.UpdateStatus(pdb); err == nil {
			break
		}
	}

	return err
}

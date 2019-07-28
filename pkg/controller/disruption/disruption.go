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
	"time"

	apps "k8s.io/api/apps/v1beta1"
	"k8s.io/api/core/v1"
	"k8s.io/api/extensions/v1beta1"
	policy "k8s.io/api/policy/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	appsv1informers "k8s.io/client-go/informers/apps/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	policyinformers "k8s.io/client-go/informers/policy/v1beta1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	policyclientset "k8s.io/client-go/kubernetes/typed/policy/v1beta1"
	appsv1listers "k8s.io/client-go/listers/apps/v1"
	corelisters "k8s.io/client-go/listers/core/v1"
	policylisters "k8s.io/client-go/listers/policy/v1beta1"
	scaleclient "k8s.io/client-go/scale"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller"

	"k8s.io/klog"
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
	mapper     apimeta.RESTMapper

	scaleNamespacer scaleclient.ScalesGetter

	pdbLister       policylisters.PodDisruptionBudgetLister
	pdbListerSynced cache.InformerSynced

	podLister       corelisters.PodLister
	podListerSynced cache.InformerSynced

	rcLister       corelisters.ReplicationControllerLister
	rcListerSynced cache.InformerSynced

	rsLister       appsv1listers.ReplicaSetLister
	rsListerSynced cache.InformerSynced

	dLister       appsv1listers.DeploymentLister
	dListerSynced cache.InformerSynced

	ssLister       appsv1listers.StatefulSetLister
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
type podControllerFinder func(controllerRef *metav1.OwnerReference, namespace string) (*controllerAndScale, error)

func NewDisruptionController(
	podInformer coreinformers.PodInformer,
	pdbInformer policyinformers.PodDisruptionBudgetInformer,
	rcInformer coreinformers.ReplicationControllerInformer,
	rsInformer appsv1informers.ReplicaSetInformer,
	dInformer appsv1informers.DeploymentInformer,
	ssInformer appsv1informers.StatefulSetInformer,
	kubeClient clientset.Interface,
	restMapper apimeta.RESTMapper,
	scaleNamespacer scaleclient.ScalesGetter,
) *DisruptionController {
	dc := &DisruptionController{
		kubeClient:   kubeClient,
		queue:        workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "disruption"),
		recheckQueue: workqueue.NewNamedDelayingQueue("disruption_recheck"),
		broadcaster:  record.NewBroadcaster(),
	}
	dc.recorder = dc.broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "controllermanager"})

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

	dc.mapper = restMapper
	dc.scaleNamespacer = scaleNamespacer

	return dc
}

// The workload resources do implement the scale subresource, so it would
// be possible to only check the scale subresource here. But since there is no
// way to take advantage of listers with scale subresources, we use the workload
// resources directly and only fall back to the scale subresource when needed.
func (dc *DisruptionController) finders() []podControllerFinder {
	return []podControllerFinder{dc.getPodReplicationController, dc.getPodDeployment, dc.getPodReplicaSet,
		dc.getPodStatefulSet, dc.getScaleController}
}

var (
	controllerKindRS  = v1beta1.SchemeGroupVersion.WithKind("ReplicaSet")
	controllerKindSS  = apps.SchemeGroupVersion.WithKind("StatefulSet")
	controllerKindRC  = v1.SchemeGroupVersion.WithKind("ReplicationController")
	controllerKindDep = v1beta1.SchemeGroupVersion.WithKind("Deployment")
)

// getPodReplicaSet finds a replicaset which has no matching deployments.
func (dc *DisruptionController) getPodReplicaSet(controllerRef *metav1.OwnerReference, namespace string) (*controllerAndScale, error) {
	ok, err := verifyGroupKind(controllerRef, controllerKindRS.Kind, []string{"apps", "extensions"})
	if !ok || err != nil {
		return nil, err
	}
	rs, err := dc.rsLister.ReplicaSets(namespace).Get(controllerRef.Name)
	if err != nil {
		// The only possible error is NotFound, which is ok here.
		return nil, nil
	}
	if rs.UID != controllerRef.UID {
		return nil, nil
	}
	controllerRef = metav1.GetControllerOf(rs)
	if controllerRef != nil && controllerRef.Kind == controllerKindDep.Kind {
		// Skip RS if it's controlled by a Deployment.
		return nil, nil
	}
	return &controllerAndScale{rs.UID, *(rs.Spec.Replicas)}, nil
}

// getPodStatefulSet returns the statefulset referenced by the provided controllerRef.
func (dc *DisruptionController) getPodStatefulSet(controllerRef *metav1.OwnerReference, namespace string) (*controllerAndScale, error) {
	ok, err := verifyGroupKind(controllerRef, controllerKindSS.Kind, []string{"apps"})
	if !ok || err != nil {
		return nil, err
	}
	ss, err := dc.ssLister.StatefulSets(namespace).Get(controllerRef.Name)
	if err != nil {
		// The only possible error is NotFound, which is ok here.
		return nil, nil
	}
	if ss.UID != controllerRef.UID {
		return nil, nil
	}

	return &controllerAndScale{ss.UID, *(ss.Spec.Replicas)}, nil
}

// getPodDeployments finds deployments for any replicasets which are being managed by deployments.
func (dc *DisruptionController) getPodDeployment(controllerRef *metav1.OwnerReference, namespace string) (*controllerAndScale, error) {
	ok, err := verifyGroupKind(controllerRef, controllerKindRS.Kind, []string{"apps", "extensions"})
	if !ok || err != nil {
		return nil, err
	}
	rs, err := dc.rsLister.ReplicaSets(namespace).Get(controllerRef.Name)
	if err != nil {
		// The only possible error is NotFound, which is ok here.
		return nil, nil
	}
	if rs.UID != controllerRef.UID {
		return nil, nil
	}
	controllerRef = metav1.GetControllerOf(rs)
	if controllerRef == nil {
		return nil, nil
	}

	ok, err = verifyGroupKind(controllerRef, controllerKindDep.Kind, []string{"apps", "extensions"})
	if !ok || err != nil {
		return nil, err
	}
	deployment, err := dc.dLister.Deployments(rs.Namespace).Get(controllerRef.Name)
	if err != nil {
		// The only possible error is NotFound, which is ok here.
		return nil, nil
	}
	if deployment.UID != controllerRef.UID {
		return nil, nil
	}
	return &controllerAndScale{deployment.UID, *(deployment.Spec.Replicas)}, nil
}

func (dc *DisruptionController) getPodReplicationController(controllerRef *metav1.OwnerReference, namespace string) (*controllerAndScale, error) {
	ok, err := verifyGroupKind(controllerRef, controllerKindRC.Kind, []string{""})
	if !ok || err != nil {
		return nil, err
	}
	rc, err := dc.rcLister.ReplicationControllers(namespace).Get(controllerRef.Name)
	if err != nil {
		// The only possible error is NotFound, which is ok here.
		return nil, nil
	}
	if rc.UID != controllerRef.UID {
		return nil, nil
	}
	return &controllerAndScale{rc.UID, *(rc.Spec.Replicas)}, nil
}

func (dc *DisruptionController) getScaleController(controllerRef *metav1.OwnerReference, namespace string) (*controllerAndScale, error) {
	gv, err := schema.ParseGroupVersion(controllerRef.APIVersion)
	if err != nil {
		return nil, err
	}

	gk := schema.GroupKind{
		Group: gv.Group,
		Kind:  controllerRef.Kind,
	}

	mapping, err := dc.mapper.RESTMapping(gk, gv.Version)
	if err != nil {
		return nil, err
	}
	gr := mapping.Resource.GroupResource()

	scale, err := dc.scaleNamespacer.Scales(namespace).Get(gr, controllerRef.Name)
	if err != nil {
		if errors.IsNotFound(err) {
			return nil, nil
		}
		return nil, err
	}
	if scale.UID != controllerRef.UID {
		return nil, nil
	}
	return &controllerAndScale{scale.UID, scale.Spec.Replicas}, nil
}

func verifyGroupKind(controllerRef *metav1.OwnerReference, expectedKind string, expectedGroups []string) (bool, error) {
	gv, err := schema.ParseGroupVersion(controllerRef.APIVersion)
	if err != nil {
		return false, err
	}

	if controllerRef.Kind != expectedKind {
		return false, nil
	}

	for _, group := range expectedGroups {
		if group == gv.Group {
			return true, nil
		}
	}

	return false, nil
}

func (dc *DisruptionController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer dc.queue.ShutDown()

	klog.Infof("Starting disruption controller")
	defer klog.Infof("Shutting down disruption controller")

	if !controller.WaitForCacheSync("disruption", stopCh, dc.podListerSynced, dc.pdbListerSynced, dc.rcListerSynced, dc.rsListerSynced, dc.dListerSynced, dc.ssListerSynced) {
		return
	}

	if dc.kubeClient != nil {
		klog.Infof("Sending events to api server.")
		dc.broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: dc.kubeClient.CoreV1().Events("")})
	} else {
		klog.Infof("No api server defined - no events will be sent to API server.")
	}
	go wait.Until(dc.worker, time.Second, stopCh)
	go wait.Until(dc.recheckWorker, time.Second, stopCh)

	<-stopCh
}

func (dc *DisruptionController) addDb(obj interface{}) {
	pdb := obj.(*policy.PodDisruptionBudget)
	klog.V(4).Infof("add DB %q", pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) updateDb(old, cur interface{}) {
	// TODO(mml) ignore updates where 'old' is equivalent to 'cur'.
	pdb := cur.(*policy.PodDisruptionBudget)
	klog.V(4).Infof("update DB %q", pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) removeDb(obj interface{}) {
	pdb := obj.(*policy.PodDisruptionBudget)
	klog.V(4).Infof("remove DB %q", pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) addPod(obj interface{}) {
	pod := obj.(*v1.Pod)
	klog.V(4).Infof("addPod called on pod %q", pod.Name)
	pdb := dc.getPdbForPod(pod)
	if pdb == nil {
		klog.V(4).Infof("No matching pdb for pod %q", pod.Name)
		return
	}
	klog.V(4).Infof("addPod %q -> PDB %q", pod.Name, pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) updatePod(old, cur interface{}) {
	pod := cur.(*v1.Pod)
	klog.V(4).Infof("updatePod called on pod %q", pod.Name)
	pdb := dc.getPdbForPod(pod)
	if pdb == nil {
		klog.V(4).Infof("No matching pdb for pod %q", pod.Name)
		return
	}
	klog.V(4).Infof("updatePod %q -> PDB %q", pod.Name, pdb.Name)
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
			klog.Errorf("Couldn't get object from tombstone %+v", obj)
			return
		}
		pod, ok = tombstone.Obj.(*v1.Pod)
		if !ok {
			klog.Errorf("Tombstone contained object that is not a pod %+v", obj)
			return
		}
	}
	klog.V(4).Infof("deletePod called on pod %q", pod.Name)
	pdb := dc.getPdbForPod(pod)
	if pdb == nil {
		klog.V(4).Infof("No matching pdb for pod %q", pod.Name)
		return
	}
	klog.V(4).Infof("deletePod %q -> PDB %q", pod.Name, pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) enqueuePdb(pdb *policy.PodDisruptionBudget) {
	key, err := controller.KeyFunc(pdb)
	if err != nil {
		klog.Errorf("Couldn't get key for PodDisruptionBudget object %+v: %v", pdb, err)
		return
	}
	dc.queue.Add(key)
}

func (dc *DisruptionController) enqueuePdbForRecheck(pdb *policy.PodDisruptionBudget, delay time.Duration) {
	key, err := controller.KeyFunc(pdb)
	if err != nil {
		klog.Errorf("Couldn't get key for PodDisruptionBudget object %+v: %v", pdb, err)
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
		klog.V(4).Infof("No PodDisruptionBudgets found for pod %v, PodDisruptionBudget controller will avoid syncing.", pod.Name)
		return nil
	}

	if len(pdbs) > 1 {
		msg := fmt.Sprintf("Pod %q/%q matches multiple PodDisruptionBudgets.  Chose %q arbitrarily.", pod.Namespace, pod.Name, pdbs[0].Name)
		klog.Warning(msg)
		dc.recorder.Event(pod, v1.EventTypeWarning, "MultiplePodDisruptionBudgets", msg)
	}
	return pdbs[0]
}

// This function returns pods using the PodDisruptionBudget object.
// IMPORTANT NOTE : the returned pods should NOT be modified.
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
	return pods, nil
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
		klog.V(4).Infof("Finished syncing PodDisruptionBudget %q (%v)", key, time.Since(startTime))
	}()

	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	pdb, err := dc.pdbLister.PodDisruptionBudgets(namespace).Get(name)
	if errors.IsNotFound(err) {
		klog.V(4).Infof("PodDisruptionBudget %q has been deleted", key)
		return nil
	}
	if err != nil {
		return err
	}

	if err := dc.trySync(pdb); err != nil {
		klog.Errorf("Failed to sync pdb %s/%s: %v", pdb.Namespace, pdb.Name, err)
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
		dc.recorder.Eventf(pdb, v1.EventTypeWarning, "CalculateExpectedPodCountFailed", "Failed to calculate the number of expected pods: %v", err)
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

	if pdb.Spec.MaxUnavailable != nil {
		expectedCount, err = dc.getExpectedScale(pdb, pods)
		if err != nil {
			return
		}
		var maxUnavailable int
		maxUnavailable, err = intstr.GetValueFromIntOrPercent(pdb.Spec.MaxUnavailable, int(expectedCount), true)
		if err != nil {
			return
		}
		desiredHealthy = expectedCount - int32(maxUnavailable)
		if desiredHealthy < 0 {
			desiredHealthy = 0
		}
	} else if pdb.Spec.MinAvailable != nil {
		if pdb.Spec.MinAvailable.Type == intstr.Int {
			desiredHealthy = pdb.Spec.MinAvailable.IntVal
			expectedCount = int32(len(pods))
		} else if pdb.Spec.MinAvailable.Type == intstr.String {
			expectedCount, err = dc.getExpectedScale(pdb, pods)
			if err != nil {
				return
			}

			var minAvailable int
			minAvailable, err = intstr.GetValueFromIntOrPercent(pdb.Spec.MinAvailable, int(expectedCount), true)
			if err != nil {
				return
			}
			desiredHealthy = int32(minAvailable)
		}
	}
	return
}

func (dc *DisruptionController) getExpectedScale(pdb *policy.PodDisruptionBudget, pods []*v1.Pod) (expectedCount int32, err error) {
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

	// 1. Find the controller for each pod.  If any pod has 0 controllers,
	// that's an error. With ControllerRef, a pod can only have 1 controller.
	for _, pod := range pods {
		controllerRef := metav1.GetControllerOf(pod)
		if controllerRef == nil {
			err = fmt.Errorf("found no controller ref for pod %q", pod.Name)
			dc.recorder.Event(pdb, v1.EventTypeWarning, "NoControllerRef", err.Error())
			return
		}

		// If we already know the scale of the controller there is no need to do anything.
		if _, found := controllerScale[controllerRef.UID]; found {
			continue
		}

		// Check all the supported controllers to find the desired scale.
		foundController := false
		for _, finder := range dc.finders() {
			var controllerNScale *controllerAndScale
			controllerNScale, err = finder(controllerRef, pod.Namespace)
			if err != nil {
				return
			}
			if controllerNScale != nil {
				controllerScale[controllerNScale.UID] = controllerNScale.scale
				foundController = true
				break
			}
		}
		if !foundController {
			err = fmt.Errorf("found no controllers for pod %q", pod.Name)
			dc.recorder.Event(pdb, v1.EventTypeWarning, "NoControllers", err.Error())
			return
		}
	}

	// 2. Add up all the controllers.
	expectedCount = 0
	for _, count := range controllerScale {
		expectedCount += count
	}

	return
}

func countHealthyPods(pods []*v1.Pod, disruptedPods map[string]metav1.Time, currentTime time.Time) (currentHealthy int32) {
Pod:
	for _, pod := range pods {
		// Pod is being deleted.
		if pod.DeletionTimestamp != nil {
			continue
		}
		// Pod is expected to be deleted soon.
		if disruptionTime, found := disruptedPods[pod.Name]; found && disruptionTime.Time.Add(DeletionTimeout).After(currentTime) {
			continue
		}
		if podutil.IsPodReady(pod) {
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
			klog.V(1).Infof("Pod %s/%s was expected to be deleted at %s but it wasn't, updating pdb %s/%s",
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
	newPdb := pdb.DeepCopy()
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
		apiequality.Semantic.DeepEqual(pdb.Status.DisruptedPods, disruptedPods) &&
		pdb.Status.ObservedGeneration == pdb.Generation {
		return nil
	}

	newPdb := pdb.DeepCopy()
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
	}
	return pdb

}

func (dc *DisruptionController) writePdbStatus(pdb *policy.PodDisruptionBudget) error {
	pdbClient := dc.kubeClient.PolicyV1beta1().PodDisruptionBudgets(pdb.Namespace)
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

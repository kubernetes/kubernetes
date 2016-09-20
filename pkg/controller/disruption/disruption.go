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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/record"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"

	"github.com/golang/glog"
)

const statusUpdateRetries = 2

type updater func(*policy.PodDisruptionBudget) error

type DisruptionController struct {
	kubeClient *client.Client

	pdbStore      cache.Store
	pdbController *framework.Controller
	pdbLister     cache.StoreToPodDisruptionBudgetLister

	podController framework.ControllerInterface
	podLister     cache.StoreToPodLister

	rcIndexer    cache.Indexer
	rcController *framework.Controller
	rcLister     cache.StoreToReplicationControllerLister

	rsStore      cache.Store
	rsController *framework.Controller
	rsLister     cache.StoreToReplicaSetLister

	dIndexer    cache.Indexer
	dController *framework.Controller
	dLister     cache.StoreToDeploymentLister

	queue *workqueue.Type

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
type podControllerFinder func(*api.Pod) ([]controllerAndScale, error)

func NewDisruptionController(podInformer framework.SharedIndexInformer, kubeClient *client.Client) *DisruptionController {
	dc := &DisruptionController{
		kubeClient:    kubeClient,
		podController: podInformer.GetController(),
		queue:         workqueue.NewNamed("disruption"),
		broadcaster:   record.NewBroadcaster(),
	}
	dc.recorder = dc.broadcaster.NewRecorder(api.EventSource{Component: "controllermanager"})

	dc.getUpdater = func() updater { return dc.writePdbStatus }

	dc.podLister.Indexer = podInformer.GetIndexer()

	podInformer.AddEventHandler(framework.ResourceEventHandlerFuncs{
		AddFunc:    dc.addPod,
		UpdateFunc: dc.updatePod,
		DeleteFunc: dc.deletePod,
	})

	dc.pdbStore, dc.pdbController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return dc.kubeClient.Policy().PodDisruptionBudgets(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return dc.kubeClient.Policy().PodDisruptionBudgets(api.NamespaceAll).Watch(options)
			},
		},
		&policy.PodDisruptionBudget{},
		30*time.Second,
		framework.ResourceEventHandlerFuncs{
			AddFunc:    dc.addDb,
			UpdateFunc: dc.updateDb,
			DeleteFunc: dc.removeDb,
		},
	)
	dc.pdbLister.Store = dc.pdbStore

	dc.rcIndexer, dc.rcController = framework.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return dc.kubeClient.ReplicationControllers(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return dc.kubeClient.ReplicationControllers(api.NamespaceAll).Watch(options)
			},
		},
		&api.ReplicationController{},
		30*time.Second,
		framework.ResourceEventHandlerFuncs{},
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	dc.rcLister.Indexer = dc.rcIndexer

	dc.rsStore, dc.rsController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return dc.kubeClient.Extensions().ReplicaSets(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return dc.kubeClient.Extensions().ReplicaSets(api.NamespaceAll).Watch(options)
			},
		},
		&extensions.ReplicaSet{},
		30*time.Second,
		framework.ResourceEventHandlerFuncs{},
	)

	dc.rsLister.Store = dc.rsStore

	dc.dIndexer, dc.dController = framework.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return dc.kubeClient.Extensions().Deployments(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return dc.kubeClient.Extensions().Deployments(api.NamespaceAll).Watch(options)
			},
		},
		&extensions.Deployment{},
		30*time.Second,
		framework.ResourceEventHandlerFuncs{},
		cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc},
	)

	dc.dLister.Indexer = dc.dIndexer

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
	return []podControllerFinder{dc.getPodReplicationControllers, dc.getPodDeployments, dc.getPodReplicaSets}
}

// getPodReplicaSets finds replicasets which have no matching deployments.
func (dc *DisruptionController) getPodReplicaSets(pod *api.Pod) ([]controllerAndScale, error) {
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
		_, err := dc.dLister.GetDeploymentsForReplicaSet(&rs)
		if err == nil { // A deployment was found, so this finder will not count this RS.
			continue
		}
		controllerScale[rs.UID] = rs.Spec.Replicas
	}

	for uid, scale := range controllerScale {
		cas = append(cas, controllerAndScale{UID: uid, scale: scale})
	}

	return cas, nil
}

// getPodDeployments finds deployments for any replicasets which are being managed by deployments.
func (dc *DisruptionController) getPodDeployments(pod *api.Pod) ([]controllerAndScale, error) {
	cas := []controllerAndScale{}
	rss, err := dc.rsLister.GetPodReplicaSets(pod)
	// GetPodReplicaSets returns an error only if no ReplicaSets are found.  We
	// don't return that as an error to the caller.
	if err != nil {
		return cas, nil
	}
	controllerScale := map[types.UID]int32{}
	for _, rs := range rss {
		ds, err := dc.dLister.GetDeploymentsForReplicaSet(&rs)
		// GetDeploymentsForReplicaSet returns an error only if no matching
		// deployments are found.  In that case we skip this ReplicaSet.
		if err != nil {
			continue
		}
		for _, d := range ds {
			controllerScale[d.UID] = d.Spec.Replicas
		}
	}

	for uid, scale := range controllerScale {
		cas = append(cas, controllerAndScale{UID: uid, scale: scale})
	}

	return cas, nil
}

func (dc *DisruptionController) getPodReplicationControllers(pod *api.Pod) ([]controllerAndScale, error) {
	cas := []controllerAndScale{}
	rcs, err := dc.rcLister.GetPodControllers(pod)
	if err == nil {
		for _, rc := range rcs {
			cas = append(cas, controllerAndScale{UID: rc.UID, scale: rc.Spec.Replicas})
		}
	}
	return cas, nil
}

func (dc *DisruptionController) Run(stopCh <-chan struct{}) {
	glog.V(0).Infof("Starting disruption controller")
	if dc.kubeClient != nil {
		glog.V(0).Infof("Sending events to api server.")
		dc.broadcaster.StartRecordingToSink(dc.kubeClient.Events(""))
	} else {
		glog.V(0).Infof("No api server defined - no events will be sent to API server.")
	}
	go dc.pdbController.Run(stopCh)
	go dc.podController.Run(stopCh)
	go dc.rcController.Run(stopCh)
	go dc.rsController.Run(stopCh)
	go dc.dController.Run(stopCh)
	go wait.Until(dc.worker, time.Second, stopCh)
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
	pod := obj.(*api.Pod)
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
	pod := cur.(*api.Pod)
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
	pod, ok := obj.(*api.Pod)
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
		pod, ok = tombstone.Obj.(*api.Pod)
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

func (dc *DisruptionController) getPdbForPod(pod *api.Pod) *policy.PodDisruptionBudget {
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
		dc.recorder.Event(pod, api.EventTypeWarning, "MultiplePodDisruptionBudgets", msg)
	}
	return &pdbs[0]
}

func (dc *DisruptionController) getPodsForPdb(pdb *policy.PodDisruptionBudget) ([]*api.Pod, error) {
	sel, err := unversioned.LabelSelectorAsSelector(pdb.Spec.Selector)
	if sel.Empty() {
		return []*api.Pod{}, nil
	}
	if err != nil {
		return []*api.Pod{}, err
	}
	pods, err := dc.podLister.Pods(pdb.Namespace).List(sel)
	if err != nil {
		return []*api.Pod{}, err
	}
	// TODO: Do we need to copy here?
	result := make([]*api.Pod, 0, len(pods))
	for i := range pods {
		result = append(result, &(*pods[i]))
	}
	return result, nil
}

func (dc *DisruptionController) worker() {
	work := func() bool {
		key, quit := dc.queue.Get()
		if quit {
			return quit
		}
		defer dc.queue.Done(key)
		glog.V(4).Infof("Syncing PodDisruptionBudget %q", key.(string))
		if err := dc.sync(key.(string)); err != nil {
			glog.Errorf("Error syncing PodDisruptionBudget %v, requeuing: %v", key.(string), err)
			// TODO(mml): In order to be safe in the face of a total inability to write state
			// changes, we should write an expiration timestamp here and consumers
			// of the PDB state (the /evict subresource handler) should check that
			// any 'true' state is relatively fresh.

			// TODO(mml): file an issue to that effect

			// TODO(mml): If we used a workqueue.RateLimitingInterface, we could
			// improve our behavior (be a better citizen) when we need to retry.
			dc.queue.Add(key)
		}
		return false
	}
	for {
		if quit := work(); quit {
			return
		}
	}
}

func (dc *DisruptionController) sync(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing PodDisruptionBudget %q (%v)", key, time.Now().Sub(startTime))
	}()

	obj, exists, err := dc.pdbLister.Store.GetByKey(key)
	if !exists {
		return err
	}
	if err != nil {
		glog.Errorf("unable to retrieve PodDisruptionBudget %v from store: %v", key, err)
		return err
	}

	pdb := obj.(*policy.PodDisruptionBudget)

	if err := dc.trySync(pdb); err != nil {
		return dc.failSafe(pdb)
	}

	return nil
}

func (dc *DisruptionController) trySync(pdb *policy.PodDisruptionBudget) error {
	pods, err := dc.getPodsForPdb(pdb)
	if err != nil {
		return err
	}

	expectedCount, desiredHealthy, err := dc.getExpectedPodCount(pdb, pods)
	if err != nil {
		return err
	}

	currentHealthy := countHealthyPods(pods)
	err = dc.updatePdbSpec(pdb, currentHealthy, desiredHealthy, expectedCount)

	return err
}

func (dc *DisruptionController) getExpectedPodCount(pdb *policy.PodDisruptionBudget, pods []*api.Pod) (expectedCount, desiredHealthy int32, err error) {
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
				dc.recorder.Event(pdb, api.EventTypeWarning, "NoControllers", err.Error())
				return
			} else if controllerCount > 1 {
				err = fmt.Errorf("pod %q has %v>1 controllers", pod.Name, controllerCount)
				dc.recorder.Event(pdb, api.EventTypeWarning, "TooManyControllers", err.Error())
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

func countHealthyPods(pods []*api.Pod) (currentHealthy int32) {
Pod:
	for _, pod := range pods {
		for _, c := range pod.Status.Conditions {
			if c.Type == api.PodReady && c.Status == api.ConditionTrue {
				currentHealthy++
				continue Pod
			}
		}
	}

	return
}

// failSafe is an attempt to at least update the PodDisruptionAllowed field to
// false if everything something else has failed.  This is one place we
// implement the  "fail open" part of the design since if we manage to update
// this field correctly, we will prevent the /evict handler from approving an
// eviction when it may be unsafe to do so.
func (dc *DisruptionController) failSafe(pdb *policy.PodDisruptionBudget) error {
	obj, err := api.Scheme.DeepCopy(*pdb)
	if err != nil {
		return err
	}
	newPdb := obj.(policy.PodDisruptionBudget)
	newPdb.Status.PodDisruptionAllowed = false

	return dc.getUpdater()(&newPdb)
}

func (dc *DisruptionController) updatePdbSpec(pdb *policy.PodDisruptionBudget, currentHealthy, desiredHealthy, expectedCount int32) error {
	// We require expectedCount to be > 0 so that PDBs which currently match no
	// pods are in a safe state when their first pods appear but this controller
	// has not updated their status yet.  This isn't the only race, but it's a
	// common one that's easy to detect.
	disruptionAllowed := currentHealthy-1 >= desiredHealthy && expectedCount > 0

	if pdb.Status.CurrentHealthy == currentHealthy && pdb.Status.DesiredHealthy == desiredHealthy && pdb.Status.ExpectedPods == expectedCount && pdb.Status.PodDisruptionAllowed == disruptionAllowed {
		return nil
	}

	obj, err := api.Scheme.DeepCopy(*pdb)
	if err != nil {
		return err
	}
	newPdb := obj.(policy.PodDisruptionBudget)

	newPdb.Status = policy.PodDisruptionBudgetStatus{
		CurrentHealthy:       currentHealthy,
		DesiredHealthy:       desiredHealthy,
		ExpectedPods:         expectedCount,
		PodDisruptionAllowed: disruptionAllowed,
	}

	return dc.getUpdater()(&newPdb)
}

// refresh tries to re-GET the given PDB.  If there are any errors, it just
// returns the old PDB.  Intended to be used in a retry loop where it runs a
// bounded number of times.
func refresh(pdbClient client.PodDisruptionBudgetInterface, pdb *policy.PodDisruptionBudget) *policy.PodDisruptionBudget {
	newPdb, err := pdbClient.Get(pdb.Name)
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

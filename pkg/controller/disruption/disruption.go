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

	dStore      cache.Store
	dController *framework.Controller
	dLister     cache.StoreToDeploymentLister

	queue *workqueue.Type

	getUpdater func() updater
}

// This is used to return (controller, scale) pairs from the controller finder
// functions.
type controllerAndScale struct {
	types.UID
	scale int32
}

// This is a simple function interface that maps a pod to a list of controllers
// and their scale.
type podControllerFinder func(*api.Pod) ([]controllerAndScale, error)

func NewDisruptionController(podInformer framework.SharedIndexInformer, kubeClient *client.Client) *DisruptionController {
	dc := &DisruptionController{
		kubeClient:    kubeClient,
		podController: podInformer.GetController(),
		queue:         workqueue.New(),
	}

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

	dc.dStore, dc.dController = framework.NewInformer(
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
	)

	dc.dLister.Store = dc.dStore

	return dc
}

// FIXME(mml): Also cover the case where there is a ReplicaSet with no deployment.
func (dc *DisruptionController) finders() []podControllerFinder {
	return []podControllerFinder{dc.getPodReplicationControllers, dc.getPodDeployments}
}

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
		// GetDeploymentsForReplicaSet returns an error only if no matching
		// deployments are found.  We don't return that as an error to the caller.
		ds, err := dc.dLister.GetDeploymentsForReplicaSet(&rs)
		if err != nil {
			return cas, nil
		}
		for _, d := range ds {
			// FIXME(mml): use ScaleStatus
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
			// FIXME(mml): use ScaleStatus
			cas = append(cas, controllerAndScale{UID: rc.UID, scale: rc.Spec.Replicas})
		}
	}
	return cas, nil
}

func (dc *DisruptionController) Run(stopCh <-chan struct{}) {
	glog.Infof("Starting disruption controller")
	go dc.pdbController.Run(stopCh)
	go dc.podController.Run(stopCh)
	go dc.rcController.Run(stopCh)
	go dc.rsController.Run(stopCh)
	go dc.dController.Run(stopCh)
	go wait.Until(dc.worker, time.Second, stopCh)
	<-stopCh
	glog.Infof("Shutting down disruption controller")
}

func (dc *DisruptionController) addDb(obj interface{}) {
	pdb := obj.(*policy.PodDisruptionBudget)
	glog.Infof("add DB %q", pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) updateDb(old, cur interface{}) {
	pdb := cur.(*policy.PodDisruptionBudget)
	glog.Infof("update DB %q", pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) removeDb(obj interface{}) {
	pdb := obj.(*policy.PodDisruptionBudget)
	glog.Infof("remove DB %q", pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) addPod(obj interface{}) {
	pod := obj.(*api.Pod)
	glog.Infof("addPod called on pod %q", pod.Name)
	pdb := dc.getPdbForPod(pod)
	if pdb == nil {
		glog.Infof("No matching pdb for pod %q", pod.Name)
		return
	}
	glog.Infof("addPod %q -> PDB %q", pod.Name, pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) updatePod(old, cur interface{}) {
	pod := cur.(*api.Pod)
	glog.Infof("updatePod called on pod %q", pod.Name)
	pdb := dc.getPdbForPod(pod)
	if pdb == nil {
		glog.Infof("No matching pdb for pod %q", pod.Name)
		return
	}
	glog.Infof("updatePod %q -> PDB %q", pod.Name, pdb.Name)
	dc.enqueuePdb(pdb)
}

func (dc *DisruptionController) deletePod(obj interface{}) {
	pod := obj.(*api.Pod)
	glog.Infof("deletePod called on pod %q", pod.Name)
	pdb := dc.getPdbForPod(pod)
	if pdb == nil {
		glog.Infof("No matching pdb for pod %q", pod.Name)
		return
	}
	glog.Infof("deletePod %q -> PDB %q", pod.Name, pdb.Name)
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
		glog.Infof("No PodDisruptionBudgets found for pod %v, PodDisruptionBudget controller will avoid syncing.", pod.Name)
		return nil
	}

	// FIXME(mml): when >1 matches, emit an event and log something
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
	podList, err := dc.podLister.Pods(pdb.Namespace).List(sel)
	if err != nil {
		return []*api.Pod{}, err
	}
	pods := []*api.Pod{}
	for _, p := range podList.Items {
		obj, err := api.Scheme.DeepCopy(p)
		if err != nil {
			return []*api.Pod{}, err
		}
		pod := obj.(api.Pod)
		pods = append(pods, &pod)
	}
	return pods, nil
}

func (dc *DisruptionController) worker() {
	for {
		func() {
			key, quit := dc.queue.Get()
			if quit {
				return
			}
			defer dc.queue.Done(key)
			glog.Infof("Syncing PodDisruptionBudget %q", key.(string))
			if err := dc.sync(key.(string)); err != nil {
				glog.Errorf("Error syncing PodDisruptionBudget %v, requeuing: %v", key.(string), err)
				// TODO(mml): In order to be safe in the face of a total inability to write state
				// changes, we should write an expiration timestamp here and consumers
				// of the PDB state (the /evict subresource handler) should check that
				// any 'true' state is relatively fresh.

				// TODO(mml): file an issue to that effect
				dc.queue.Add(key)
			}
		}()
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
		glog.Errorf("Unable to retrieve PodDisruptionBudget %v from store: %v", key, err)
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

	currentHealthy := dc.countHealthyPods(pdb, pods)
	err = dc.updatePdbSpec(pdb, currentHealthy, desiredHealthy, expectedCount)

	return err
}

// FIXME(mml): explain *that* and *why* we are using ScaleStatus here, once we are.
func (dc *DisruptionController) getExpectedPodCount(pdb *policy.PodDisruptionBudget, pods []*api.Pod) (expectedCount, desiredHealthy int32, err error) {
	err = nil
	if pdb.Spec.MinAvailable.Type == intstr.Int {
		desiredHealthy = pdb.Spec.MinAvailable.IntVal
		expectedCount = int32(len(pods))
	} else if pdb.Spec.MinAvailable.Type == intstr.String {
		// When the user specifies a fraction of pods that must be available, we
		// use as the fraction's denominator the sum of the values of the /scale
		// subresource for the pods' controllers.

		// A mapping from controllers to their scale.
		controllerScale := map[types.UID]int32{}

		// 1. Find the controller(s) for each pod.  If any pod has 0 controllers,
		// that's an error.
		for _, pod := range pods {
			foundController := false
			for _, finder := range dc.finders() {
				var controllers []controllerAndScale
				controllers, err = finder(pod)
				if err != nil {
					return
				}
				if len(controllers) > 0 {
					foundController = true
					for _, controller := range controllers {
						controllerScale[controller.UID] = controller.scale
					}
				}
			}
			if !foundController {
				err = fmt.Errorf("Asked for percentage, but found no controllers for pod %q", pod.Name)
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

func (dc *DisruptionController) countHealthyPods(pdb *policy.PodDisruptionBudget, pods []*api.Pod) (currentHealthy int32) {
	currentHealthy = 0

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
	disruptionAllowed := currentHealthy >= desiredHealthy && expectedCount > 0

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

func (dc *DisruptionController) writePdbStatus(pdb *policy.PodDisruptionBudget) error {
	pdbClient := dc.kubeClient.Policy().PodDisruptionBudgets(pdb.Namespace)

	// TODO(mml): retry
	_, err := pdbClient.UpdateStatus(pdb)
	return err
}

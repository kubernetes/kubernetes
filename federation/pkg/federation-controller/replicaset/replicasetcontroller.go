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

package replicaset

import (
	"encoding/json"
	"reflect"
	"time"

	"github.com/golang/glog"

	fed "k8s.io/kubernetes/federation/apis/federation"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_4"
	planner "k8s.io/kubernetes/federation/pkg/federation-controller/replicaset/planner"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	extensionsv1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_4"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	FedReplicaSetPreferencesAnnotation = "federation.kubernetes.io/replica-set-preferences"
	allClustersKey                     = "THE_ALL_CLUSTER_KEY"
	UserAgentName                      = "Federation-replicaset-Controller"
)

var (
	replicaSetReviewDelay    = 10 * time.Second
	clusterAvailableDelay    = 20 * time.Second
	clusterUnavailableDelay  = 60 * time.Second
	allReplicaSetReviewDealy = 2 * time.Minute
)

func parseFederationReplicaSetReference(frs *extensionsv1.ReplicaSet) (*fed.FederatedReplicaSetPreferences, error) {
	if frs.Annotations == nil {
		return nil, nil
	}
	frsPrefString, found := frs.Annotations[FedReplicaSetPreferencesAnnotation]
	if !found {
		return nil, nil
	}
	var frsPref fed.FederatedReplicaSetPreferences
	if err := json.Unmarshal([]byte(frsPrefString), &frsPref); err != nil {
		return nil, err
	}
	return &frsPref, nil
}

type ReplicaSetController struct {
	fedClient fedclientset.Interface

	replicaSetController *framework.Controller
	replicaSetStore      cache.StoreToReplicaSetLister

	fedReplicaSetInformer fedutil.FederatedInformer
	fedPodInformer        fedutil.FederatedInformer

	replicasetDeliverer *fedutil.DelayingDeliverer
	clusterDeliverer    *fedutil.DelayingDeliverer
	replicasetWorkQueue workqueue.Interface

	replicaSetBackoff *flowcontrol.Backoff

	defaultPlanner *planner.Planner
}

// NewclusterController returns a new cluster controller
func NewReplicaSetController(federationClient fedclientset.Interface) *ReplicaSetController {
	frsc := &ReplicaSetController{
		fedClient:           federationClient,
		replicasetDeliverer: fedutil.NewDelayingDeliverer(),
		clusterDeliverer:    fedutil.NewDelayingDeliverer(),
		replicasetWorkQueue: workqueue.New(),
		replicaSetBackoff:   flowcontrol.NewBackOff(5*time.Second, time.Minute),
		defaultPlanner: planner.NewPlanner(&fed.FederatedReplicaSetPreferences{
			Clusters: map[string]fed.ClusterReplicaSetPreferences{
				"*": {Weight: 1},
			},
		}),
	}

	replicaSetFedInformerFactory := func(cluster *fedv1.Cluster, clientset kubeclientset.Interface) (cache.Store, framework.ControllerInterface) {
		return framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					return clientset.Extensions().ReplicaSets(apiv1.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return clientset.Extensions().ReplicaSets(apiv1.NamespaceAll).Watch(options)
				},
			},
			&extensionsv1.ReplicaSet{},
			controller.NoResyncPeriodFunc(),
			fedutil.NewTriggerOnAllChanges(
				func(obj runtime.Object) { frsc.deliverLocalReplicaSet(obj, allReplicaSetReviewDealy) },
			),
		)
	}
	clusterLifecycle := fedutil.ClusterLifecycleHandlerFuncs{
		ClusterAvailable: func(cluster *fedv1.Cluster) {
			frsc.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterUnavailableDelay)
		},
		ClusterUnavailable: func(cluster *fedv1.Cluster, _ []interface{}) {
			frsc.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterUnavailableDelay)
		},
	}
	frsc.fedReplicaSetInformer = fedutil.NewFederatedInformer(federationClient, replicaSetFedInformerFactory, &clusterLifecycle)

	podFedInformerFactory := func(cluster *fedv1.Cluster, clientset kubeclientset.Interface) (cache.Store, framework.ControllerInterface) {
		return framework.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					return clientset.Core().Pods(apiv1.NamespaceAll).List(options)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					return clientset.Core().Pods(apiv1.NamespaceAll).Watch(options)
				},
			},
			&apiv1.Pod{},
			controller.NoResyncPeriodFunc(),
			fedutil.NewTriggerOnAllChanges(
				func(obj runtime.Object) {
					frsc.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterUnavailableDelay)
				},
			),
		)
	}
	frsc.fedPodInformer = fedutil.NewFederatedInformer(federationClient, podFedInformerFactory, &fedutil.ClusterLifecycleHandlerFuncs{})

	frsc.replicaSetStore.Store, frsc.replicaSetController = framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return frsc.fedClient.Extensions().ReplicaSets(apiv1.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return frsc.fedClient.Extensions().ReplicaSets(apiv1.NamespaceAll).Watch(options)
			},
		},
		&extensionsv1.ReplicaSet{},
		controller.NoResyncPeriodFunc(),
		fedutil.NewTriggerOnMetaAndSpecChanges(
			func(obj runtime.Object) { frsc.deliverFedReplicaSetObj(obj, replicaSetReviewDelay) },
		),
	)

	return frsc
}

func (frsc *ReplicaSetController) Run(workers int, stopCh <-chan struct{}) {
	go frsc.replicaSetController.Run(stopCh)
	frsc.fedReplicaSetInformer.Start()
	frsc.fedPodInformer.Start()

	frsc.replicasetDeliverer.StartWithHandler(func(item *fedutil.DelayingDelivererItem) {
		frsc.replicasetWorkQueue.Add(item.Key)
	})
	frsc.clusterDeliverer.StartWithHandler(func(_ *fedutil.DelayingDelivererItem) {
		frsc.reconcileReplicaSetsOnClusterChange()
	})

	for !frsc.isSynced() {
		time.Sleep(5 * time.Millisecond)
	}

	for i := 0; i < workers; i++ {
		go wait.Until(frsc.worker, time.Second, stopCh)
	}

	go func() {
		select {
		case <-time.After(time.Minute):
			frsc.replicaSetBackoff.GC()
		case <-stopCh:
			return
		}
	}()

	<-stopCh
	glog.Infof("Shutting down ReplicaSetController")
	frsc.replicasetDeliverer.Stop()
	frsc.clusterDeliverer.Stop()
	frsc.replicasetWorkQueue.ShutDown()
	frsc.fedReplicaSetInformer.Stop()
	frsc.fedPodInformer.Stop()
}

func (frsc *ReplicaSetController) isSynced() bool {
	if !frsc.fedReplicaSetInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	clusters, err := frsc.fedReplicaSetInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !frsc.fedReplicaSetInformer.GetTargetStore().ClustersSynced(clusters) {
		return false
	}

	if !frsc.fedPodInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	if !frsc.fedPodInformer.GetTargetStore().ClustersSynced(clusters) {
		return false
	}

	if !frsc.replicaSetController.HasSynced() {
		glog.V(2).Infof("federation replicaset list not synced")
		return false
	}
	return true
}

func (frsc *ReplicaSetController) deliverLocalReplicaSet(obj interface{}, duration time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %v: %v", obj, err)
		return
	}
	_, exists, err := frsc.replicaSetStore.Store.GetByKey(key)
	if err != nil {
		glog.Errorf("Couldn't get federation replicaset %v: %v", key, err)
		return
	}
	if exists { // ignore replicasets exists only in local k8s
		frsc.deliverReplicaSetByKey(key, duration, false)
	}
}

func (frsc *ReplicaSetController) deliverFedReplicaSetObj(obj interface{}, delay time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	frsc.deliverReplicaSetByKey(key, delay, false)
}

func (frsc *ReplicaSetController) deliverReplicaSetByKey(key string, delay time.Duration, failed bool) {
	if failed {
		frsc.replicaSetBackoff.Next(key, time.Now())
		delay = delay + frsc.replicaSetBackoff.Get(key)
	} else {
		frsc.replicaSetBackoff.Reset(key)
	}
	frsc.replicasetDeliverer.DeliverAfter(key, nil, delay)
}

func (frsc *ReplicaSetController) worker() {
	for {
		item, quit := frsc.replicasetWorkQueue.Get()
		if quit {
			return
		}
		key := item.(string)
		err := frsc.reconcileReplicaSet(key)
		frsc.replicasetWorkQueue.Done(item)
		if err != nil {
			glog.Errorf("Error syncing cluster controller: %v", err)
			frsc.deliverReplicaSetByKey(key, 0, true)
		}
	}
}

func (frsc *ReplicaSetController) schedule(frs *extensionsv1.ReplicaSet, clusters []*fedv1.Cluster,
	current map[string]int64, estimatedCapacity map[string]int64) map[string]int64 {
	// TODO: integrate real scheduler

	plnr := frsc.defaultPlanner
	frsPref, err := parseFederationReplicaSetReference(frs)
	if err != nil {
		glog.Info("Invalid ReplicaSet specific preference, use default. rs: %v, err: %v", frs, err)
	}
	if frsPref != nil { // create a new planner if user specified a preference
		plnr = planner.NewPlanner(frsPref)
	}

	replicas := int64(*frs.Spec.Replicas)
	var clusterNames []string
	for _, cluster := range clusters {
		clusterNames = append(clusterNames, cluster.Name)
	}
	scheduleResult, overflow := plnr.Plan(replicas, clusterNames, current, estimatedCapacity)
	// make sure the return contains clusters need to zero the replicas
	result := make(map[string]int64)
	for clusterName := range current {
		result[clusterName] = 0
	}
	for clusterName, replicas := range scheduleResult {
		result[clusterName] = replicas
	}
	for clusterName, replicas := range overflow {
		result[clusterName] += replicas
	}
	return result
}

func (frsc *ReplicaSetController) reconcileReplicaSet(key string) error {
	if !frsc.isSynced() {
		frsc.deliverReplicaSetByKey(key, clusterAvailableDelay, false)
		return nil
	}

	glog.Infof("Start reconcile replicaset %q", key)
	startTime := time.Now()
	defer glog.Infof("Finished reconcile replicaset %q (%v)", key, time.Now().Sub(startTime))

	obj, exists, err := frsc.replicaSetStore.Store.GetByKey(key)
	if err != nil {
		return err
	}
	if !exists {
		// don't delete local replicasets for now
		return nil
	}
	frs := obj.(*extensionsv1.ReplicaSet)

	clusters, err := frsc.fedReplicaSetInformer.GetReadyClusters()
	if err != nil {
		return err
	}

	// collect current status and do schedule
	allPods, err := frsc.fedPodInformer.GetTargetStore().List()
	if err != nil {
		return err
	}
	podStatus, err := AnalysePods(frs, allPods, time.Now())
	current := make(map[string]int64)
	estimatedCapacity := make(map[string]int64)
	for _, cluster := range clusters {
		lrsObj, exists, err := frsc.fedReplicaSetInformer.GetTargetStore().GetByKey(cluster.Name, key)
		if err != nil {
			return err
		}
		if exists {
			lrs := lrsObj.(*extensionsv1.ReplicaSet)
			current[cluster.Name] = int64(podStatus[cluster.Name].RunningAndReady) // include pending as well?
			unschedulable := int64(podStatus[cluster.Name].Unschedulable)
			if unschedulable > 0 {
				estimatedCapacity[cluster.Name] = int64(*lrs.Spec.Replicas) - unschedulable
			}
		}
	}

	scheduleResult := frsc.schedule(frs, clusters, current, estimatedCapacity)

	glog.Infof("Start syncing local replicaset %v", scheduleResult)

	fedStatus := extensionsv1.ReplicaSetStatus{ObservedGeneration: frs.Generation}
	for clusterName, replicas := range scheduleResult {
		// TODO: updater or parallelizer doesnn't help as results are needed for updating fed rs status
		clusterClient, err := frsc.fedReplicaSetInformer.GetClientsetForCluster(clusterName)
		if err != nil {
			return err
		}
		lrsObj, exists, err := frsc.fedReplicaSetInformer.GetTargetStore().GetByKey(clusterName, key)
		if err != nil {
			return err
		} else if !exists {
			if replicas > 0 {
				lrs := &extensionsv1.ReplicaSet{
					ObjectMeta: apiv1.ObjectMeta{
						Name:        frs.Name,
						Namespace:   frs.Namespace,
						Labels:      frs.Labels,
						Annotations: frs.Annotations,
					},
					Spec: frs.Spec,
				}
				specReplicas := int32(replicas)
				lrs.Spec.Replicas = &specReplicas
				lrs, err = clusterClient.Extensions().ReplicaSets(frs.Namespace).Create(lrs)
				if err != nil {
					return err
				}
				fedStatus.Replicas += lrs.Status.Replicas
				fedStatus.FullyLabeledReplicas += lrs.Status.FullyLabeledReplicas
			}
		} else {
			lrs := lrsObj.(*extensionsv1.ReplicaSet)
			lrsExpectedSpec := frs.Spec
			specReplicas := int32(replicas)
			lrsExpectedSpec.Replicas = &specReplicas
			if !reflect.DeepEqual(lrs.Spec, lrsExpectedSpec) {
				lrs.Spec = lrsExpectedSpec
				lrs, err = clusterClient.Extensions().ReplicaSets(frs.Namespace).Update(lrs)
				if err != nil {
					return err
				}
			}
			fedStatus.Replicas += lrs.Status.Replicas
			fedStatus.FullyLabeledReplicas += lrs.Status.FullyLabeledReplicas
			// leave the replicaset even the replicas dropped to 0
		}
	}
	if fedStatus.Replicas != frs.Status.Replicas || fedStatus.FullyLabeledReplicas != frs.Status.FullyLabeledReplicas {
		frs.Status = fedStatus
		_, err = frsc.fedClient.Extensions().ReplicaSets(frs.Namespace).UpdateStatus(frs)
		if err != nil {
			return err
		}
	}

	return nil
}

func (frsc *ReplicaSetController) reconcileReplicaSetsOnClusterChange() {
	if !frsc.isSynced() {
		frsc.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterAvailableDelay)
	}
	rss := frsc.replicaSetStore.Store.List()
	for _, rs := range rss {
		key, _ := controller.KeyFunc(rs)
		frsc.deliverReplicaSetByKey(key, 0, false)
	}
}

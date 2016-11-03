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

package job

import (
	"encoding/json"
	"reflect"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/golang/glog"

	fed "k8s.io/kubernetes/federation/apis/federation"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_release_1_5"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/planner"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	batchv1 "k8s.io/kubernetes/pkg/apis/batch/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	FedJobPreferencesAnnotation = "federation.kubernetes.io/job-preferences"
	allClustersKey              = "THE_ALL_CLUSTER_KEY"
	UserAgentName               = "Federation-Job-Controller"
)

var (
	jobReviewDelay          = 10 * time.Second
	clusterAvailableDelay   = 20 * time.Second
	clusterUnavailableDelay = 60 * time.Second
	allJobReviewDelay       = 2 * time.Minute
	updateTimeout           = 30 * time.Second
	backoffInitial          = 5 * time.Second
	backoffMax              = 1 * time.Minute
)

func parseFedJobReference(frs *batchv1.Job) (*fed.FederatedReplicaSetPreferences, error) {
	if frs.Annotations == nil {
		return nil, nil
	}
	frsPrefString, found := frs.Annotations[FedJobPreferencesAnnotation]
	if !found {
		return nil, nil
	}
	var frsPref fed.FederatedReplicaSetPreferences
	if err := json.Unmarshal([]byte(frsPrefString), &frsPref); err != nil {
		return nil, err
	}
	return &frsPref, nil
}

type JobController struct {
	fedClient fedclientset.Interface

	jobController *cache.Controller
	jobStore      cache.Store

	fedJobInformer fedutil.FederatedInformer

	jobDeliverer     *fedutil.DelayingDeliverer
	clusterDeliverer *fedutil.DelayingDeliverer
	jobWorkQueue     workqueue.Interface
	// For updating members of federation.
	fedUpdater fedutil.FederatedUpdater

	jobBackoff *flowcontrol.Backoff
	// For events
	eventRecorder record.EventRecorder

	defaultPlanner *planner.Planner
}

func NewJobController(fedClient fedclientset.Interface) *JobController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(fedClient))
	recorder := broadcaster.NewRecorder(api.EventSource{Component: "federated-job-controller"})

	fjc := &JobController{
		fedClient:        fedClient,
		jobDeliverer:     fedutil.NewDelayingDeliverer(),
		clusterDeliverer: fedutil.NewDelayingDeliverer(),
		jobWorkQueue:     workqueue.New(),
		jobBackoff:       flowcontrol.NewBackOff(backoffInitial, backoffMax),
		defaultPlanner: planner.NewPlanner(&fed.FederatedReplicaSetPreferences{
			Clusters: map[string]fed.ClusterReplicaSetPreferences{
				"*": {Weight: 1},
			},
		}),
		eventRecorder: recorder,
	}

	jobFedInformerFactory := func(cluster *fedv1.Cluster, clientset kubeclientset.Interface) (cache.Store, cache.ControllerInterface) {
		return cache.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options api.ListOptions) (runtime.Object, error) {
					versionedOptions := fedutil.VersionizeV1ListOptions(options)
					return clientset.Batch().Jobs(apiv1.NamespaceAll).List(versionedOptions)
				},
				WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
					versionedOptions := fedutil.VersionizeV1ListOptions(options)
					return clientset.Batch().Jobs(apiv1.NamespaceAll).Watch(versionedOptions)
				},
			},
			&batchv1.Job{},
			controller.NoResyncPeriodFunc(),
			fedutil.NewTriggerOnAllChanges(
				func(obj runtime.Object) { fjc.deliverLocalJob(obj, jobReviewDelay) },
			),
		)
	}
	clusterLifecycle := fedutil.ClusterLifecycleHandlerFuncs{
		ClusterAvailable: func(cluster *fedv1.Cluster) {
			fjc.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterAvailableDelay)
		},
		ClusterUnavailable: func(cluster *fedv1.Cluster, _ []interface{}) {
			fjc.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterUnavailableDelay)
		},
	}
	fjc.fedJobInformer = fedutil.NewFederatedInformer(fedClient, jobFedInformerFactory, &clusterLifecycle)

	fjc.jobStore, fjc.jobController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				versionedOptions := fedutil.VersionizeV1ListOptions(options)
				return fjc.fedClient.Batch().Jobs(apiv1.NamespaceAll).List(versionedOptions)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				versionedOptions := fedutil.VersionizeV1ListOptions(options)
				return fjc.fedClient.Batch().Jobs(apiv1.NamespaceAll).Watch(versionedOptions)
			},
		},
		&batchv1.Job{},
		controller.NoResyncPeriodFunc(),
		fedutil.NewTriggerOnMetaAndSpecChanges(
			func(obj runtime.Object) { fjc.deliverFedJobObj(obj, 0) },
		),
	)

	fjc.fedUpdater = fedutil.NewFederatedUpdater(fjc.fedJobInformer,
		func(client kubeclientset.Interface, obj runtime.Object) error {
			rs := obj.(*batchv1.Job)
			_, err := client.Batch().Jobs(rs.Namespace).Create(rs)
			return err
		},
		func(client kubeclientset.Interface, obj runtime.Object) error {
			rs := obj.(*batchv1.Job)
			_, err := client.Batch().Jobs(rs.Namespace).Update(rs)
			return err
		},
		func(client kubeclientset.Interface, obj runtime.Object) error {
			rs := obj.(*batchv1.Job)
			err := client.Batch().Jobs(rs.Namespace).Delete(rs.Name, &apiv1.DeleteOptions{})
			return err
		})

	return fjc
}

func (fjc *JobController) Run(workers int, stopCh <-chan struct{}) {
	go fjc.jobController.Run(stopCh)
	fjc.fedJobInformer.Start()

	fjc.jobDeliverer.StartWithHandler(func(item *fedutil.DelayingDelivererItem) {
		fjc.jobWorkQueue.Add(item.Key)
	})
	fjc.clusterDeliverer.StartWithHandler(func(_ *fedutil.DelayingDelivererItem) {
		fjc.reconcileJobsOnClusterChange()
	})

	for !fjc.isSynced() {
		time.Sleep(5 * time.Millisecond)
	}

	for i := 0; i < workers; i++ {
		go wait.Until(fjc.worker, time.Second, stopCh)
	}

	fedutil.StartBackoffGC(fjc.jobBackoff, stopCh)

	<-stopCh
	glog.Infof("Shutting down JobController")
	fjc.jobDeliverer.Stop()
	fjc.clusterDeliverer.Stop()
	fjc.jobWorkQueue.ShutDown()
	fjc.fedJobInformer.Stop()
}

func (fjc *JobController) isSynced() bool {
	if !fjc.fedJobInformer.ClustersSynced() {
		glog.V(3).Infof("Cluster list not synced")
		return false
	}
	clusters, err := fjc.fedJobInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if !fjc.fedJobInformer.GetTargetStore().ClustersSynced(clusters) {
		glog.V(2).Infof("cluster job list not synced")
		return false
	}

	if !fjc.jobController.HasSynced() {
		glog.V(2).Infof("federation job list not synced")
		return false
	}
	return true
}

func (fjc *JobController) deliverLocalJob(obj interface{}, duration time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %v: %v", obj, err)
		return
	}
	_, exists, err := fjc.jobStore.GetByKey(key)
	if err != nil {
		glog.Errorf("Couldn't get federated job %v: %v", key, err)
		return
	}
	if exists { // ignore jobs exists only in local k8s
		fjc.deliverJobByKey(key, duration, false)
	}
}

func (fjc *JobController) deliverFedJobObj(obj interface{}, delay time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	fjc.deliverJobByKey(key, delay, false)
}

func (fjc *JobController) deliverJobByKey(key string, delay time.Duration, failed bool) {
	if failed {
		fjc.jobBackoff.Next(key, time.Now())
		delay = delay + fjc.jobBackoff.Get(key)
	} else {
		fjc.jobBackoff.Reset(key)
	}
	fjc.jobDeliverer.DeliverAfter(key, nil, delay)
}

type reconciliationStatus string

const (
	statusAllOk       = reconciliationStatus("ALL_OK")
	statusNeedRecheck = reconciliationStatus("RECHECK")
	statusError       = reconciliationStatus("ERROR")
	statusNotSynced   = reconciliationStatus("NOSYNC")
)

func (fjc *JobController) worker() {
	for {
		item, quit := fjc.jobWorkQueue.Get()
		if quit {
			return
		}
		key := item.(string)
		status, err := fjc.reconcileJob(key)
		fjc.jobWorkQueue.Done(item)
		if err != nil {
			glog.Errorf("Error syncing job controller: %v", err)
			fjc.deliverJobByKey(key, 0, true)
		} else {
			switch status {
			case statusAllOk:
				break
			case statusError:
				fjc.deliverJobByKey(key, 0, true)
			case statusNeedRecheck:
				fjc.deliverJobByKey(key, jobReviewDelay, false)
			case statusNotSynced:
				fjc.deliverJobByKey(key, clusterAvailableDelay, false)
			default:
				glog.Errorf("Unhandled reconciliation status: %s", status)
				fjc.deliverJobByKey(key, jobReviewDelay, false)
			}
		}
	}
}

type scheduleResult struct {
	Parallelism *int32
	Completions *int32
}

func (fjc *JobController) schedule(fjob *batchv1.Job, clusters []*fedv1.Cluster) map[string]scheduleResult {
	plnr := fjc.defaultPlanner
	frsPref, err := parseFedJobReference(fjob)
	if err != nil {
		glog.Warningf("Invalid job specific preference, use default. rs: %v, err: %v", fjob, err)
	}
	if frsPref != nil { // create a new planner if user specified a preference
		plnr = planner.NewPlanner(frsPref)
	}

	parallelism := int64(*fjob.Spec.Parallelism)
	var clusterNames []string
	for _, cluster := range clusters {
		clusterNames = append(clusterNames, cluster.Name)
	}
	parallelismResult, _ := plnr.Plan(parallelism, clusterNames, nil, nil, fjob.Namespace+"/"+fjob.Name)

	if frsPref != nil {
		for _, clusterPref := range frsPref.Clusters {
			clusterPref.MinReplicas = 0
			clusterPref.MaxReplicas = nil
		}
		plnr = planner.NewPlanner(frsPref)
	}
	clusterNames = nil
	for clusterName := range parallelismResult {
		clusterNames = append(clusterNames, clusterName)
	}
	completionsResult := make(map[string]int64)
	if fjob.Spec.Completions != nil {
		completionsResult, _ = plnr.Plan(int64(*fjob.Spec.Completions), clusterNames, nil, nil, fjob.Namespace+"/"+fjob.Name)
	}

	results := make(map[string]scheduleResult)
	for _, clusterName := range clusterNames {
		paralle := int32(parallelismResult[clusterName])
		complet := int32(completionsResult[clusterName])
		result := scheduleResult{
			Parallelism: &paralle,
		}
		if fjob.Spec.Completions != nil {
			result.Completions = &complet
		}
		results[clusterName] = result
	}

	return results
}

func (fjc *JobController) reconcileJob(key string) (reconciliationStatus, error) {
	if !fjc.isSynced() {
		return statusNotSynced, nil
	}

	glog.V(4).Infof("Start reconcile job %q", key)
	startTime := time.Now()
	defer glog.V(4).Infof("Finished reconcile job %q (%v)", key, time.Now().Sub(startTime))

	obj, exists, err := fjc.jobStore.GetByKey(key)
	if err != nil {
		return statusError, err
	}
	if !exists {
		err = fjc.deleteJob(key)
		if err != nil {
			return statusError, err
		} else {
			return statusAllOk, nil
		}
	}
	fjob := obj.(*batchv1.Job)

	clusters, err := fjc.fedJobInformer.GetReadyClusters()
	if err != nil {
		return statusError, err
	}

	scheduleResult := fjc.schedule(fjob, clusters)
	glog.V(3).Infof("Start syncing local job %s: %s\n", key, spew.Sprintf("%v", scheduleResult))

	fedStatus := batchv1.JobStatus{}
	var fedStatusFailedCondition *batchv1.JobCondition
	var fedStatusCompleteCondition *batchv1.JobCondition
	operations := make([]fedutil.FederatedOperation, 0)
	for clusterName, result := range scheduleResult {
		ljobObj, exists, err := fjc.fedJobInformer.GetTargetStore().GetByKey(clusterName, key)
		if err != nil {
			return statusError, err
		}
		ljob := &batchv1.Job{
			ObjectMeta: fedutil.CopyObjectMeta(fjob.ObjectMeta),
			Spec:       fjob.Spec,
		}
		// use selector generated at federation level, or user specified value
		manualSelector := true
		ljob.Spec.ManualSelector = &manualSelector
		ljob.Spec.Parallelism = result.Parallelism
		ljob.Spec.Completions = result.Completions

		if !exists {
			if *ljob.Spec.Parallelism > 0 /*&& (ljob.Spec.Completions == nil || *ljob.Spec.Completions > 0)*/ {
				fjc.eventRecorder.Eventf(fjob, api.EventTypeNormal, "CreateInCluster", "Creating job in cluster %s", clusterName)
				operations = append(operations, fedutil.FederatedOperation{
					Type:        fedutil.OperationTypeAdd,
					Obj:         ljob,
					ClusterName: clusterName,
				})
			}
		} else {
			currentLjob := ljobObj.(*batchv1.Job)

			// Update existing job, if needed.
			if !fedutil.ObjectMetaAndSpecEquivalent(ljob, currentLjob) {
				fjc.eventRecorder.Eventf(fjob, api.EventTypeNormal, "UpdateInCluster", "Updating job in cluster %s", clusterName)
				operations = append(operations, fedutil.FederatedOperation{
					Type:        fedutil.OperationTypeUpdate,
					Obj:         ljob,
					ClusterName: clusterName,
				})
			}

			// collect local job status
			for _, condition := range currentLjob.Status.Conditions {
				if condition.Type == batchv1.JobComplete {
					if fedStatusCompleteCondition == nil ||
						fedStatusCompleteCondition.LastTransitionTime.Before(condition.LastTransitionTime) {
						fedStatusCompleteCondition = &condition
					}
				} else if condition.Type == batchv1.JobFailed {
					if fedStatusFailedCondition == nil ||
						fedStatusFailedCondition.LastTransitionTime.Before(condition.LastTransitionTime) {
						fedStatusFailedCondition = &condition
					}
				}
			}
			if currentLjob.Status.StartTime != nil {
				if fedStatus.StartTime == nil || fedStatus.StartTime.After(currentLjob.Status.StartTime.Time) {
					fedStatus.StartTime = currentLjob.Status.StartTime
				}
			}
			if currentLjob.Status.CompletionTime != nil {
				if fedStatus.CompletionTime == nil || fedStatus.CompletionTime.Before(*currentLjob.Status.CompletionTime) {
					fedStatus.CompletionTime = currentLjob.Status.CompletionTime
				}
			}
			fedStatus.Active += currentLjob.Status.Active
			fedStatus.Succeeded += currentLjob.Status.Succeeded
			fedStatus.Failed += currentLjob.Status.Failed
		}
	}

	// federated job fails if any local job failes
	if fedStatusFailedCondition != nil {
		fedStatus.Conditions = append(fedStatus.Conditions, *fedStatusFailedCondition)
	} else if fedStatusCompleteCondition != nil {
		fedStatus.Conditions = append(fedStatus.Conditions, *fedStatusCompleteCondition)
	}
	if !reflect.DeepEqual(fedStatus, fjob.Status) {
		fjob.Status = fedStatus
		_, err = fjc.fedClient.Batch().Jobs(fjob.Namespace).UpdateStatus(fjob)
		if err != nil {
			return statusError, err
		}
	}

	if len(operations) == 0 {
		// Everything is in order
		return statusAllOk, nil
	}

	if glog.V(4) {
		for i, op := range operations {
			job := op.Obj.(*batchv1.Job)
			glog.V(4).Infof("operation[%d]: %s, %s/%s/%s, %d", i, op.Type, op.ClusterName, job.Namespace, job.Name, *job.Spec.Parallelism)
		}
	}
	err = fjc.fedUpdater.UpdateWithOnError(operations, updateTimeout, func(op fedutil.FederatedOperation, operror error) {
		fjc.eventRecorder.Eventf(fjob, api.EventTypeNormal, "FailedUpdateInCluster", "Job update in cluster %s failed: %v", op.ClusterName, operror)
		glog.Errorf("Failed to execute updates for %s %s/%s: %v", op.Type, op.ClusterName, key, operror)
	})
	if err != nil {
		return statusError, err
	}

	// Some operations were made, reconcile after a while.
	return statusNeedRecheck, nil

}

func (fjc *JobController) reconcileJobsOnClusterChange() {
	if !fjc.isSynced() {
		fjc.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterAvailableDelay)
	}
	jobs := fjc.jobStore.List()
	for _, job := range jobs {
		key, _ := controller.KeyFunc(job)
		fjc.deliverJobByKey(key, 0, false)
	}
}

func (frsc *JobController) deleteJob(key string) error {
	glog.V(3).Infof("deleting job: %v", key)
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}
	// try delete from all clusters
	clusters, err := frsc.fedJobInformer.GetReadyClusters()
	for _, cluster := range clusters {
		clusterClient, err := frsc.fedJobInformer.GetClientsetForCluster(cluster.Name)
		if err != nil {
			return err
		}
		err = clusterClient.Batch().Jobs(namespace).Delete(name, &apiv1.DeleteOptions{})
		if err != nil && !errors.IsNotFound(err) {
			glog.Warningf("failed deleting repicaset %v/%v/%v, err: %v", cluster.Name, namespace, name, err)
			return err
		}
	}
	return nil

}

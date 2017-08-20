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
	"fmt"
	"reflect"
	"time"

	"github.com/davecgh/go-spew/spew"
	"github.com/golang/glog"

	batchv1 "k8s.io/api/batch/v1"
	clientv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	kubeclientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/flowcontrol"
	"k8s.io/client-go/util/workqueue"
	fed "k8s.io/kubernetes/federation/apis/federation"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/planner"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/replicapreferences"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/controller"
)

const (
	fedJobPreferencesAnnotation = "federation.kubernetes.io/job-preferences"
	allClustersKey              = "THE_ALL_CLUSTER_KEY"
	// UserAgentName is the user agent used in the federation client
	UserAgentName = "Federation-Job-Controller"
	// ControllerName is name of this controller
	ControllerName = "jobs"
)

var (
	// RequiredResources is the resource group version of the type this controller manages
	RequiredResources       = []schema.GroupVersionResource{batchv1.SchemeGroupVersion.WithResource("jobs")}
	jobReviewDelay          = 10 * time.Second
	clusterAvailableDelay   = 20 * time.Second
	clusterUnavailableDelay = 60 * time.Second
	updateTimeout           = 30 * time.Second
	backoffInitial          = 5 * time.Second
	backoffMax              = 1 * time.Minute
)

// FederationJobController synchronizes the state of a federated job object
// to clusters that are members of the federation.
type FederationJobController struct {
	fedClient fedclientset.Interface

	jobController cache.Controller
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
	deletionHelper *deletionhelper.DeletionHelper
}

// NewJobController creates a new federation job controller
func NewJobController(fedClient fedclientset.Interface) *FederationJobController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(fedClient))
	recorder := broadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "federated-job-controller"})
	fjc := &FederationJobController{
		fedClient:        fedClient,
		jobDeliverer:     fedutil.NewDelayingDeliverer(),
		clusterDeliverer: fedutil.NewDelayingDeliverer(),
		jobWorkQueue:     workqueue.New(),
		jobBackoff:       flowcontrol.NewBackOff(backoffInitial, backoffMax),
		defaultPlanner: planner.NewPlanner(&fed.ReplicaAllocationPreferences{
			Clusters: map[string]fed.ClusterPreferences{
				"*": {Weight: 1},
			},
		}),
		eventRecorder: recorder,
	}

	jobFedInformerFactory := func(cluster *fedv1.Cluster, clientset kubeclientset.Interface) (cache.Store, cache.Controller) {
		return cache.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
					return clientset.BatchV1().Jobs(metav1.NamespaceAll).List(options)
				},
				WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
					return clientset.BatchV1().Jobs(metav1.NamespaceAll).Watch(options)
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
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return fjc.fedClient.BatchV1().Jobs(metav1.NamespaceAll).List(options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return fjc.fedClient.BatchV1().Jobs(metav1.NamespaceAll).Watch(options)
			},
		},
		&batchv1.Job{},
		controller.NoResyncPeriodFunc(),
		fedutil.NewTriggerOnMetaAndSpecChanges(
			func(obj runtime.Object) { fjc.deliverFedJobObj(obj, 0) },
		),
	)

	fjc.fedUpdater = fedutil.NewFederatedUpdater(fjc.fedJobInformer, "job", updateTimeout, fjc.eventRecorder,
		func(client kubeclientset.Interface, obj runtime.Object) error {
			rs := obj.(*batchv1.Job)
			_, err := client.BatchV1().Jobs(rs.Namespace).Create(rs)
			return err
		},
		func(client kubeclientset.Interface, obj runtime.Object) error {
			rs := obj.(*batchv1.Job)
			_, err := client.BatchV1().Jobs(rs.Namespace).Update(rs)
			return err
		},
		func(client kubeclientset.Interface, obj runtime.Object) error {
			rs := obj.(*batchv1.Job)
			err := client.BatchV1().Jobs(rs.Namespace).Delete(rs.Name, &metav1.DeleteOptions{})
			return err
		})

	fjc.deletionHelper = deletionhelper.NewDeletionHelper(
		fjc.updateJob,
		// objNameFunc
		func(obj runtime.Object) string {
			job := obj.(*batchv1.Job)
			return job.Name
		},
		fjc.fedJobInformer,
		fjc.fedUpdater,
	)

	return fjc
}

// Sends the given updated object to apiserver.
// Assumes that the given object is a job.
func (fjc *FederationJobController) updateJob(obj runtime.Object) (runtime.Object, error) {
	job := obj.(*batchv1.Job)
	return fjc.fedClient.BatchV1().Jobs(job.Namespace).Update(job)
}

// Run starts the syncing of federation jobs to the clusters.
func (fjc *FederationJobController) Run(workers int, stopCh <-chan struct{}) {
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
	glog.Infof("Shutting down FederationJobController")
	fjc.jobDeliverer.Stop()
	fjc.clusterDeliverer.Stop()
	fjc.jobWorkQueue.ShutDown()
	fjc.fedJobInformer.Stop()
}

func (fjc *FederationJobController) isSynced() bool {
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

func (fjc *FederationJobController) deliverLocalJob(obj interface{}, duration time.Duration) {
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

func (fjc *FederationJobController) deliverFedJobObj(obj interface{}, delay time.Duration) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		glog.Errorf("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	fjc.deliverJobByKey(key, delay, false)
}

func (fjc *FederationJobController) deliverJobByKey(key string, delay time.Duration, failed bool) {
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

func (fjc *FederationJobController) worker() {
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

func (fjc *FederationJobController) schedule(fjob *batchv1.Job, clusters []*fedv1.Cluster) map[string]scheduleResult {
	plnr := fjc.defaultPlanner
	frsPref, err := replicapreferences.GetAllocationPreferences(fjob, fedJobPreferencesAnnotation)
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

func (fjc *FederationJobController) reconcileJob(key string) (reconciliationStatus, error) {
	if !fjc.isSynced() {
		return statusNotSynced, nil
	}

	glog.V(4).Infof("Start reconcile job %q", key)
	startTime := time.Now()
	defer glog.V(4).Infof("Finished reconcile job %q (%v)", key, time.Now().Sub(startTime))

	objFromStore, exists, err := fjc.jobStore.GetByKey(key)
	if err != nil {
		return statusError, err
	}
	if !exists {
		// deleted federated job, nothing need to do
		return statusAllOk, nil
	}

	// Create a copy before modifying the obj to prevent race condition with other readers of obj from store.
	obj, err := api.Scheme.DeepCopy(objFromStore)
	fjob, ok := obj.(*batchv1.Job)
	if err != nil || !ok {
		return statusError, err
	}

	// delete job
	if fjob.DeletionTimestamp != nil {
		if err := fjc.delete(fjob); err != nil {
			fjc.eventRecorder.Eventf(fjob, api.EventTypeNormal, "DeleteFailed", "Job delete failed: %v", err)
			return statusError, err
		}
		return statusAllOk, nil
	}

	glog.V(3).Infof("Ensuring delete object from underlying clusters finalizer for job: %s\n", key)
	// Add the required finalizers before creating a job in underlying clusters.
	updatedJobObj, err := fjc.deletionHelper.EnsureFinalizers(fjob)
	if err != nil {
		return statusError, err
	}
	fjob = updatedJobObj.(*batchv1.Job)

	clusters, err := fjc.fedJobInformer.GetReadyClusters()
	if err != nil {
		return statusError, err
	}

	scheduleResult := fjc.schedule(fjob, clusters)
	glog.V(3).Infof("Start syncing local job %s: %s\n", key, spew.Sprintf("%v", scheduleResult))

	fedStatus := batchv1.JobStatus{}
	var fedStatusFailedCondition *batchv1.JobCondition
	var fedStatusCompleteCondition *batchv1.JobCondition
	var operations []fedutil.FederatedOperation
	for clusterName, result := range scheduleResult {
		ljobObj, exists, err := fjc.fedJobInformer.GetTargetStore().GetByKey(clusterName, key)
		if err != nil {
			return statusError, err
		}
		ljob := &batchv1.Job{
			ObjectMeta: fedutil.DeepCopyRelevantObjectMeta(fjob.ObjectMeta),
			Spec:       *fedutil.DeepCopyApiTypeOrPanic(&fjob.Spec).(*batchv1.JobSpec),
		}
		// use selector generated at federation level, or user specified value
		manualSelector := true
		ljob.Spec.ManualSelector = &manualSelector
		ljob.Spec.Parallelism = result.Parallelism
		ljob.Spec.Completions = result.Completions

		if !exists {
			if *ljob.Spec.Parallelism > 0 {
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
						fedStatusCompleteCondition.LastTransitionTime.Before(&condition.LastTransitionTime) {
						fedStatusCompleteCondition = &condition
					}
				} else if condition.Type == batchv1.JobFailed {
					if fedStatusFailedCondition == nil ||
						fedStatusFailedCondition.LastTransitionTime.Before(&condition.LastTransitionTime) {
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
				if fedStatus.CompletionTime == nil || fedStatus.CompletionTime.Before(currentLjob.Status.CompletionTime) {
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
		_, err = fjc.fedClient.BatchV1().Jobs(fjob.Namespace).UpdateStatus(fjob)
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
	err = fjc.fedUpdater.Update(operations)
	if err != nil {
		return statusError, err
	}

	// Some operations were made, reconcile after a while.
	return statusNeedRecheck, nil

}

func (fjc *FederationJobController) reconcileJobsOnClusterChange() {
	if !fjc.isSynced() {
		fjc.clusterDeliverer.DeliverAfter(allClustersKey, nil, clusterAvailableDelay)
	}
	jobs := fjc.jobStore.List()
	for _, job := range jobs {
		key, _ := controller.KeyFunc(job)
		fjc.deliverJobByKey(key, 0, false)
	}
}

// delete deletes the given job or returns error if the deletion was not complete.
func (fjc *FederationJobController) delete(job *batchv1.Job) error {
	glog.V(3).Infof("Handling deletion of job: %s/%s\n", job.Namespace, job.Name)
	_, err := fjc.deletionHelper.HandleObjectInUnderlyingClusters(job)
	if err != nil {
		return err
	}

	err = fjc.fedClient.BatchV1().Jobs(job.Namespace).Delete(job.Name, nil)
	if err != nil {
		// Its all good if the error is not found error. That means it is deleted already and we do not have to do anything.
		// This is expected when we are processing an update as a result of job finalizer deletion.
		// The process that deleted the last finalizer is also going to delete the job and we do not have to do anything.
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to delete job: %s/%s, %v", job.Namespace, job.Name, err)
		}
	}
	return nil
}

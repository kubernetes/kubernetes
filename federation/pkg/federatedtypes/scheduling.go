/*
Copyright 2017 The Kubernetes Authors.

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

package federatedtypes

import (
	"bytes"
	"fmt"
	"reflect"
	"sort"
	"time"

	batchv1 "k8s.io/api/batch/v1"
	apiv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	fedapi "k8s.io/kubernetes/federation/apis/federation"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	hpautil "k8s.io/kubernetes/federation/pkg/federation-controller/util/hpa"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/planner"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/podanalyzer"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/replicapreferences"

	"github.com/golang/glog"
)

// ScheduleAction is used by the interface ScheduleObject of SchedulingAdapter
// to sync controller reconcile to convey the action type needed for the
// particular cluster local object in ScheduleObject
type ScheduleAction string

const (
	ActionAdd    = "add"
	ActionDelete = "delete"
)

// ReplicaStatus contains the details of status fields from the cluster objects,
// which need accumulation to update the status of the federated object.
type ReplicaStatus struct {
	Replicas             int32
	UpdatedReplicas      int32
	FullyLabeledReplicas int32
	ReadyReplicas        int32
	AvailableReplicas    int32
}

// ReplicaScheduleState is the result of adapter specific schedule() function,
// which is then used to update objects into clusters.
type ReplicaScheduleState struct {
	isSelected bool
	replicas   int64
}

// ReplicaSchedulingInfo wraps the information that a replica type (rs or deployment)
// SchedulingAdapter needs to update objects per a schedule.
type ReplicaSchedulingInfo struct {
	ScheduleState map[string]*ReplicaScheduleState
	Status        ReplicaStatus
}

type JobStatus struct {
	//Conditions     []JobCondition
	StartTime      *metav1.Time
	CompletionTime *metav1.Time
	Active         int32
	Succeeded      int32
	Failed         int32
}

type JobScheduleState struct {
	parallelism *int32
	completions *int32
}

// JobSchedulingInfo wraps the information that a job type SchedulingAdapter needs to
// update objects per a schedule
type JobSchedulingInfo struct {
	ScheduleState map[string]*JobScheduleState
	Status        JobStatus
}

// SchedulingAdapter defines operations for interacting with a
// federated type that requires more complex synchronization logic.
type SchedulingAdapter interface {
	GetSchedule(obj pkgruntime.Object, key string, clusters []*federationapi.Cluster, informer fedutil.FederatedInformer) (interface{}, error)
	ScheduleObject(cluster *federationapi.Cluster, clusterObj pkgruntime.Object, federationObjCopy pkgruntime.Object, schedulingInfo interface{}) (pkgruntime.Object, ScheduleAction, error)
	UpdateFederatedStatus(obj pkgruntime.Object, schedulingInfo interface{}) error

	// EquivalentIgnoringSchedule returns whether obj1 and obj2 are
	// equivalent ignoring differences due to scheduling.
	EquivalentIgnoringSchedule(obj1, obj2 pkgruntime.Object) bool
}

// replicaSchedulingAdapter is meant to be embedded in other type adapters that require
// workload scheduling with actual pod replicas.
type replicaSchedulingAdapter struct {
	preferencesAnnotationName string
	updateStatusFunc          func(pkgruntime.Object, interface{}) error
}

type jobSchedulingAdapter struct {
	preferencesAnnotationName string
	updateStatusFunc          func(pkgruntime.Object, interface{}) error
}

func (a *replicaSchedulingAdapter) IsSchedulingAdapter() bool {
	return true
}

func (a *jobSchedulingAdapter) IsSchedulingAdapter() bool {
	return true
}

func isSelected(names []string, name string) bool {
	for _, val := range names {
		if val == name {
			return true
		}
	}
	return false
}

func isObjHpaControlled(fedObj pkgruntime.Object) (bool, error) {
	hpaSelectedClusters, error := hpautil.GetHpaTargetClusterList(fedObj)
	if error != nil {
		return false, error
	}

	if hpaSelectedClusters == nil {
		return false, nil
	}
	return true, nil
}

// initializeScheduleState initializes the schedule state for consumption by schedule
// functions (schedule or simple schedule). After this initialization the state would
// already have information, if only a subset of clusters targetted by hpa, or all clusters
// need to be considered by the actual scheduling functions.
// The return bool named hpaControlled tells if this object is controlled by hpa or not.
func initializeScheduleState(fedObj pkgruntime.Object, clusterNames []string) (map[string]*ReplicaScheduleState, bool, error) {
	initialState := make(map[string]*ReplicaScheduleState)
	hpaControlled := false
	hpaSelectedClusters, error := hpautil.GetHpaTargetClusterList(fedObj)
	if error != nil {
		return nil, hpaControlled, error
	}

	if hpaSelectedClusters != nil {
		hpaControlled = true
	}
	for _, clusterName := range clusterNames {
		replicaState := ReplicaScheduleState{
			isSelected: false,
			replicas:   0,
		}
		if hpaControlled {
			if isSelected(hpaSelectedClusters.Names, clusterName) {
				replicaState.isSelected = true
			}
		}
		initialState[clusterName] = &replicaState
	}
	return initialState, hpaControlled, nil
}

func (a *replicaSchedulingAdapter) GetSchedule(obj pkgruntime.Object, key string, clusters []*federationapi.Cluster, informer fedutil.FederatedInformer) (interface{}, error) {
	var clusterNames []string
	for _, cluster := range clusters {
		clusterNames = append(clusterNames, cluster.Name)
	}

	// Schedule the pods across the existing clusters.
	objectGetter := func(clusterName, key string) (interface{}, bool, error) {
		return informer.GetTargetStore().GetByKey(clusterName, key)
	}
	podsGetter := func(clusterName string, obj pkgruntime.Object) (*apiv1.PodList, error) {
		clientset, err := informer.GetClientsetForCluster(clusterName)
		if err != nil {
			return nil, err
		}
		selectorObj := reflect.ValueOf(obj).Elem().FieldByName("Spec").FieldByName("Selector").Interface().(*metav1.LabelSelector)
		selector, err := metav1.LabelSelectorAsSelector(selectorObj)
		if err != nil {
			return nil, fmt.Errorf("invalid selector: %v", err)
		}
		metadata, err := meta.Accessor(obj)
		if err != nil {
			return nil, err
		}
		return clientset.Core().Pods(metadata.GetNamespace()).List(metav1.ListOptions{LabelSelector: selector.String()})
	}

	initializedState, hpaControlled, err := initializeScheduleState(obj, clusterNames)
	if err != nil {
		return nil, err
	}

	if hpaControlled {
		state, err := simpleSchedule(initializedState, key, objectGetter)
		if err != nil {
			return nil, err
		}
		return &ReplicaSchedulingInfo{
			ScheduleState: state,
			Status:        ReplicaStatus{},
		}, nil
	}

	currentReplicasPerCluster, estimatedCapacity, err := clustersReplicaState(clusterNames, key, objectGetter, podsGetter)
	if err != nil {
		return nil, err
	}

	fedPref, err := replicapreferences.GetAllocationPreferences(obj, a.preferencesAnnotationName)
	if err != nil {
		glog.Infof("Invalid workload-type specific preference, using default. object: %v, err: %v", obj, err)
	}
	if fedPref == nil {
		fedPref = &fedapi.ReplicaAllocationPreferences{
			Clusters: map[string]fedapi.ClusterPreferences{
				"*": {Weight: 1},
			},
		}
	}

	plnr := planner.NewPlanner(fedPref)

	return &ReplicaSchedulingInfo{
		ScheduleState: schedule(plnr, obj, key, clusterNames, currentReplicasPerCluster, estimatedCapacity, initializedState),
		Status:        ReplicaStatus{},
	}, nil
}

func (a *jobSchedulingAdapter) GetSchedule(obj pkgruntime.Object, key string, clusters []*federationapi.Cluster, informer fedutil.FederatedInformer) (interface{}, error) {
	job := obj.(*batchv1.Job)
	var clusterNames []string
	for _, cluster := range clusters {
		clusterNames = append(clusterNames, cluster.Name)
	}

	fedPref, err := replicapreferences.GetAllocationPreferences(obj, a.preferencesAnnotationName)
	if err != nil {
		glog.Infof("Invalid workload-type specific preference, using default. object: %v, err: %v", obj, err)
	}
	if fedPref == nil {
		fedPref = &fedapi.ReplicaAllocationPreferences{
			Clusters: map[string]fedapi.ClusterPreferences{
				"*": {Weight: 1},
			},
		}
	}

	plnr := planner.NewPlanner(fedPref)
	parallelismResult, _ := plnr.Plan(int64(*job.Spec.Parallelism), clusterNames, nil, nil, key)

	// Reset min/max replica values to avoid restricting the number
	// of completions allowed per cluster as it may be larger than
	// replicas specified in job preferences
	for _, clusterPref := range fedPref.Clusters {
		clusterPref.MinReplicas = 0
		clusterPref.MaxReplicas = nil
	}

	plnr = planner.NewPlanner(fedPref)

	clusterNames = nil
	for clusterName := range parallelismResult {
		clusterNames = append(clusterNames, clusterName)
	}

	completionsResult := make(map[string]int64)
	if job.Spec.Completions != nil {
		completionsResult, _ = plnr.Plan(int64(*job.Spec.Completions), clusterNames, nil, nil, key)
	}

	results := make(map[string]*JobScheduleState)
	for _, clusterName := range clusterNames {
		p := int32(parallelismResult[clusterName])
		c := int32(completionsResult[clusterName])
		scheduleInfo := JobScheduleState{
			parallelism: &p,
		}
		if job.Spec.Completions != nil {
			scheduleInfo.completions = &c
		}
		results[clusterName] = &scheduleInfo
	}

	return &JobSchedulingInfo{
		ScheduleState: results,
		Status:        JobStatus{},
	}, nil
}

func (a *replicaSchedulingAdapter) ScheduleObject(cluster *federationapi.Cluster, clusterObj pkgruntime.Object, federationObjCopy pkgruntime.Object, schedulingInfo interface{}) (pkgruntime.Object, ScheduleAction, error) {
	typedSchedulingInfo := schedulingInfo.(*ReplicaSchedulingInfo)
	clusterScheduleState := typedSchedulingInfo.ScheduleState[cluster.Name]

	if clusterObj != nil {
		schedulingStatusVal := reflect.ValueOf(typedSchedulingInfo).Elem().FieldByName("Status")
		objStatusVal := reflect.ValueOf(clusterObj).Elem().FieldByName("Status")
		for i := 0; i < schedulingStatusVal.NumField(); i++ {
			schedulingStatusField := schedulingStatusVal.Field(i)
			schedulingStatusFieldName := schedulingStatusVal.Type().Field(i).Name
			objStatusField := objStatusVal.FieldByName(schedulingStatusFieldName)
			if objStatusField.IsValid() {
				current := schedulingStatusField.Int()
				additional := objStatusField.Int()
				schedulingStatusField.SetInt(current + additional)
			}
		}
	}

	var action ScheduleAction = ""
	specReplicas := int32(0)
	// If the cluster has been selected (isSelected = true; for example by hpa)
	// and the obj does not get any replicas, then it should create one with
	// 0 replicas (which can then be scaled by hpa in that cluster).
	// On the other hand we keep the action as "unassigned" if this cluster was
	// not selected, and let the sync controller decide what to do.
	if clusterScheduleState.isSelected {
		specReplicas = int32(clusterScheduleState.replicas)
		action = ActionAdd
	}
	reflect.ValueOf(federationObjCopy).Elem().FieldByName("Spec").FieldByName("Replicas").Set(reflect.ValueOf(&specReplicas))
	return federationObjCopy, action, nil
}

func (a *jobSchedulingAdapter) ScheduleObject(cluster *federationapi.Cluster, clusterObj pkgruntime.Object, federationObjCopy pkgruntime.Object, schedulingInfo interface{}) (pkgruntime.Object, ScheduleAction, error) {
	typedSchedulingInfo := schedulingInfo.(*JobSchedulingInfo)
	clusterScheduleState := typedSchedulingInfo.ScheduleState[cluster.Name]

	if clusterObj != nil {
		schedulingStatusVal := reflect.ValueOf(typedSchedulingInfo).Elem().FieldByName("Status")
		objStatusVal := reflect.ValueOf(clusterObj).Elem().FieldByName("Status")
		for i := 0; i < schedulingStatusVal.NumField(); i++ {
			schedulingStatusField := schedulingStatusVal.Field(i)
			schedulingStatusFieldName := schedulingStatusVal.Type().Field(i).Name
			objStatusField := objStatusVal.FieldByName(schedulingStatusFieldName)
			if objStatusField.IsValid() {
				switch t := objStatusField.Interface().(type) {
				// TODO: Conditions
				case *metav1.Time:
					objTime := objStatusField.Interface().(*metav1.Time)
					if objTime == nil {
						break
					}
					if !schedulingStatusField.IsValid() ||
						(schedulingStatusFieldName == "StartTime" && schedulingStatusField.Interface().(*metav1.Time).After(objTime.Time)) ||
						(schedulingStatusFieldName == "CompletionTime" && schedulingStatusField.Interface().(*metav1.Time).Before(objTime)) {
						schedulingStatusField.Set(objStatusField)
					}
				case int32:
					current := schedulingStatusField.Int()
					additional := objStatusField.Int()
					schedulingStatusField.SetInt(current + additional)
				default:
					glog.V(2).Infof("unrecognized type %T!\n", t)
				}
			}
		}
	}

	var action ScheduleAction = ActionAdd
	manualSelector := true
	specParallelism := int32(*clusterScheduleState.parallelism)
	specCompletions := int32(*clusterScheduleState.completions)
	reflect.ValueOf(federationObjCopy).Elem().FieldByName("Spec").FieldByName("Parallelism").Set(reflect.ValueOf(&specParallelism))
	reflect.ValueOf(federationObjCopy).Elem().FieldByName("Spec").FieldByName("Completions").Set(reflect.ValueOf(&specCompletions))
	reflect.ValueOf(federationObjCopy).Elem().FieldByName("Spec").FieldByName("ManualSelector").Set(reflect.ValueOf(&manualSelector))
	return federationObjCopy, action, nil
}

func (a *replicaSchedulingAdapter) UpdateFederatedStatus(obj pkgruntime.Object, schedulingInfo interface{}) error {
	return a.updateStatusFunc(obj, schedulingInfo)
}

func (a *jobSchedulingAdapter) UpdateFederatedStatus(obj pkgruntime.Object, schedulingInfo interface{}) error {
	return a.updateStatusFunc(obj, schedulingInfo)
}

// simpleSchedule get replicas from only those clusters which are selected (by hpa scheduler).
// This aim of this is to ensure that this controller does not update objects, which are
// targetted by hpa.
func simpleSchedule(scheduleState map[string]*ReplicaScheduleState, key string, objectGetter func(clusterName string, key string) (interface{}, bool, error)) (map[string]*ReplicaScheduleState, error) {
	for clusterName, state := range scheduleState {
		// Get and consider replicas only for those clusters which are selected by hpa.
		if state.isSelected {
			obj, exists, err := objectGetter(clusterName, key)
			if err != nil {
				return nil, err
			}
			if !exists {
				continue
			}
			state.replicas = reflect.ValueOf(obj).Elem().FieldByName("Spec").FieldByName("Replicas").Elem().Int()
		}
	}

	return scheduleState, nil
}

func schedule(planner *planner.Planner, obj pkgruntime.Object, key string, clusterNames []string, currentReplicasPerCluster map[string]int64, estimatedCapacity map[string]int64, initialState map[string]*ReplicaScheduleState) map[string]*ReplicaScheduleState {
	// TODO: integrate real scheduler
	replicas := reflect.ValueOf(obj).Elem().FieldByName("Spec").FieldByName("Replicas").Elem().Int()
	scheduleResult, overflow := planner.Plan(replicas, clusterNames, currentReplicasPerCluster, estimatedCapacity, key)

	// Ensure that all current clusters end up in the scheduling result.
	// initialState, is preinitialized with all isSelected to false.
	result := initialState
	for clusterName := range currentReplicasPerCluster {
		// We consider 0 replicas equaling to no need of creating a new object.
		// isSchedule remains false in such case.
		result[clusterName].replicas = 0
	}

	for clusterName, replicas := range scheduleResult {
		result[clusterName].isSelected = true
		result[clusterName].replicas = replicas
	}
	for clusterName, replicas := range overflow {
		result[clusterName].isSelected = true
		result[clusterName].replicas += replicas
	}

	if glog.V(4) {
		buf := bytes.NewBufferString(fmt.Sprintf("Schedule - %q\n", key))
		sort.Strings(clusterNames)
		for _, clusterName := range clusterNames {
			cur := currentReplicasPerCluster[clusterName]
			target := scheduleResult[clusterName]
			fmt.Fprintf(buf, "%s: current: %d target: %d", clusterName, cur, target)
			if over, found := overflow[clusterName]; found {
				fmt.Fprintf(buf, " overflow: %d", over)
			}
			if capacity, found := estimatedCapacity[clusterName]; found {
				fmt.Fprintf(buf, " capacity: %d", capacity)
			}
			fmt.Fprintf(buf, "\n")
		}
		glog.V(4).Infof(buf.String())
	}
	return result
}

//func scheduleJobs(planner *planner.Planner, obj pkgruntime.Object, key string, clusterNames []string) map[string]*JobScheduleState {
//	// TODO: integrate real scheduler
//	parallelism := reflect.ValueOf(obj).Elem().FieldByName("Spec").FieldByName("Parallelism").Elem().Int()
//	parallelismResult, overflow := planner.Plan(parallelism, clusterNames, nil, nil, key)
//
//	// Ensure that all current clusters end up in the scheduling result.
//	result := initialState
//	for clusterName := range currentReplicasPerCluster {
//		// We consider 0 replicas equaling to no need of creating a new object.
//		// isSchedule remains false in such case.
//		result[clusterName].replicas = 0
//	}
//
//	for clusterName, replicas := range scheduleResult {
//		result[clusterName].isSelected = true
//		result[clusterName].replicas = replicas
//	}
//	for clusterName, replicas := range overflow {
//		result[clusterName].isSelected = true
//		result[clusterName].replicas += replicas
//	}
//
//	if glog.V(4) {
//		buf := bytes.NewBufferString(fmt.Sprintf("Schedule - %q\n", key))
//		sort.Strings(clusterNames)
//		for _, clusterName := range clusterNames {
//			cur := currentReplicasPerCluster[clusterName]
//			target := scheduleResult[clusterName]
//			fmt.Fprintf(buf, "%s: current: %d target: %d", clusterName, cur, target)
//			if over, found := overflow[clusterName]; found {
//				fmt.Fprintf(buf, " overflow: %d", over)
//			}
//			if capacity, found := estimatedCapacity[clusterName]; found {
//				fmt.Fprintf(buf, " capacity: %d", capacity)
//			}
//			fmt.Fprintf(buf, "\n")
//		}
//		glog.V(4).Infof(buf.String())
//	}
//	return result
//}

// clusterReplicaState returns information about the scheduling state of the pods running in the federated clusters.
func clustersReplicaState(
	clusterNames []string,
	key string,
	objectGetter func(clusterName string, key string) (interface{}, bool, error),
	podsGetter func(clusterName string, obj pkgruntime.Object) (*apiv1.PodList, error)) (currentReplicasPerCluster map[string]int64, estimatedCapacity map[string]int64, err error) {

	currentReplicasPerCluster = make(map[string]int64)
	estimatedCapacity = make(map[string]int64)

	for _, clusterName := range clusterNames {
		obj, exists, err := objectGetter(clusterName, key)
		if err != nil {
			return nil, nil, err
		}
		if !exists {
			continue
		}
		replicas := reflect.ValueOf(obj).Elem().FieldByName("Spec").FieldByName("Replicas").Elem().Int()
		readyReplicas := reflect.ValueOf(obj).Elem().FieldByName("Status").FieldByName("ReadyReplicas").Int()
		if replicas == readyReplicas {
			currentReplicasPerCluster[clusterName] = readyReplicas
		} else {
			pods, err := podsGetter(clusterName, obj.(pkgruntime.Object))
			if err != nil {
				return nil, nil, err
			}
			podStatus := podanalyzer.AnalyzePods(pods, time.Now())
			currentReplicasPerCluster[clusterName] = int64(podStatus.RunningAndReady) // include pending as well?
			unschedulable := int64(podStatus.Unschedulable)
			if unschedulable > 0 {
				estimatedCapacity[clusterName] = replicas - unschedulable
			}
		}
	}
	return currentReplicasPerCluster, estimatedCapacity, nil
}

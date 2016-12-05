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

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	fedapi "k8s.io/kubernetes/federation/apis/federation"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
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

// ReplicaSchedulingStatus contains the status of the replica type objects (rs or deployment)
// that are being scheduled into joined clusters.
type ReplicaSchedulingStatus struct {
	Replicas             int32
	UpdatedReplicas      int32
	FullyLabeledReplicas int32
	ReadyReplicas        int32
	AvailableReplicas    int32
}

// ReplicaSchedulingInfo wraps the information that a replica type (rs or deployment)
// SchedulingAdapter needs to update objects per a schedule.
type ReplicaSchedulingInfo struct {
	Schedule map[string]int64
	Status   ReplicaSchedulingStatus
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

func (a *replicaSchedulingAdapter) IsSchedulingAdapter() bool {
	return true
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
		Schedule: schedule(plnr, obj, key, clusterNames, currentReplicasPerCluster, estimatedCapacity),
		Status:   ReplicaSchedulingStatus{},
	}, nil
}

func (a *replicaSchedulingAdapter) ScheduleObject(cluster *federationapi.Cluster, clusterObj pkgruntime.Object, federationObjCopy pkgruntime.Object, schedulingInfo interface{}) (pkgruntime.Object, ScheduleAction, error) {
	typedSchedulingInfo := schedulingInfo.(*ReplicaSchedulingInfo)
	replicas, ok := typedSchedulingInfo.Schedule[cluster.Name]
	if !ok {
		replicas = 0
	}

	specReplicas := int32(replicas)
	reflect.ValueOf(federationObjCopy).Elem().FieldByName("Spec").FieldByName("Replicas").Set(reflect.ValueOf(&specReplicas))

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
	if replicas > 0 {
		action = ActionAdd
	}
	return federationObjCopy, action, nil
}

func (a *replicaSchedulingAdapter) UpdateFederatedStatus(obj pkgruntime.Object, schedulingInfo interface{}) error {
	return a.updateStatusFunc(obj, schedulingInfo)
}

func schedule(planner *planner.Planner, obj pkgruntime.Object, key string, clusterNames []string, currentReplicasPerCluster map[string]int64, estimatedCapacity map[string]int64) map[string]int64 {
	// TODO: integrate real scheduler
	replicas := reflect.ValueOf(obj).Elem().FieldByName("Spec").FieldByName("Replicas").Elem().Int()
	scheduleResult, overflow := planner.Plan(replicas, clusterNames, currentReplicasPerCluster, estimatedCapacity, key)

	// Ensure that all current clusters end up in the scheduling result.
	result := make(map[string]int64)
	for clusterName := range currentReplicasPerCluster {
		result[clusterName] = 0
	}

	for clusterName, replicas := range scheduleResult {
		result[clusterName] = replicas
	}
	for clusterName, replicas := range overflow {
		result[clusterName] += replicas
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

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
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
)

// SchedulingStatus contains the status of the objects that are being
// scheduled into joined clusters.
type SchedulingStatus struct {
	Replicas             int32
	FullyLabeledReplicas int32
	ReadyReplicas        int32
	AvailableReplicas    int32
}

// SchedulingInfo wraps the information that a SchedulingAdapter needs
// to update objects per a schedule.
type SchedulingInfo struct {
	Schedule map[string]int64
	Status   SchedulingStatus
}

// SchedulingAdapter defines operations for interacting with a
// federated type that requires more complex synchronization logic.
type SchedulingAdapter interface {
	GetSchedule(obj pkgruntime.Object, key string, clusters []*federationapi.Cluster, informer fedutil.FederatedInformer) (*SchedulingInfo, error)
	ScheduleObject(cluster *federationapi.Cluster, clusterObj pkgruntime.Object, federationObjCopy pkgruntime.Object, schedulingInfo *SchedulingInfo) (pkgruntime.Object, bool, error)
	UpdateFederatedStatus(obj pkgruntime.Object, status SchedulingStatus) error
}

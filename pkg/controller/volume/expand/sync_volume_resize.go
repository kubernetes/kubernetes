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

package expand

import (
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/controller/volume/expand/cache"
	"k8s.io/kubernetes/pkg/controller/volume/expand/util"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

type SyncVolumeResize interface {
	Run(stopCh <-chan struct{})
}

type syncResize struct {
	loopPeriod  time.Duration
	resizeMap   cache.VolumeResizeMap
	opsExecutor operationexecutor.OperationExecutor
	kubeClient  clientset.Interface
}

// NewSyncVolumeResize returns actual volume resize handler
func NewSyncVolumeResize(
	loopPeriod time.Duration,
	opsExecutor operationexecutor.OperationExecutor,
	resizeMap cache.VolumeResizeMap,
	kubeClient clientset.Interface) SyncVolumeResize {
	rc := &syncResize{
		loopPeriod:  loopPeriod,
		opsExecutor: opsExecutor,
		resizeMap:   resizeMap,
		kubeClient:  kubeClient,
	}
	return rc
}

func (rc *syncResize) Run(stopCh <-chan struct{}) {
	wait.Until(rc.Sync, rc.loopPeriod, stopCh)
}

func (rc *syncResize) Sync() {
	// Resize PVCs that require resize
	for _, pvcWithResizeRequest := range rc.resizeMap.GetPVCsWithResizeRequest() {
		uniqueVolumeKey := v1.UniqueVolumeName(pvcWithResizeRequest.UniquePVCKey())
		updatedClaim, err := markPVCResizeInProgress(pvcWithResizeRequest, rc.kubeClient)
		if err != nil {
			glog.V(5).Infof("Error setting PVC %s in progress with error : %v", pvcWithResizeRequest.QualifiedName(), err)
			continue
		}
		if updatedClaim != nil {
			pvcWithResizeRequest.PVC = updatedClaim
		}

		if rc.opsExecutor.IsOperationPending(uniqueVolumeKey, "") {
			glog.V(10).Infof("Operation for PVC %v is already pending", pvcWithResizeRequest.QualifiedName())
			continue
		}
		glog.V(5).Infof("Starting opsExecutor.ExpandVolume for volume %s", pvcWithResizeRequest.QualifiedName())
		growFuncError := rc.opsExecutor.ExpandVolume(pvcWithResizeRequest, rc.resizeMap)
		if growFuncError != nil && !exponentialbackoff.IsExponentialBackoff(growFuncError) {
			glog.Errorf("Error growing pvc %s with %v", pvcWithResizeRequest.QualifiedName(), growFuncError)
		}
		if growFuncError == nil {
			glog.V(5).Infof("Started opsExecutor.ExpandVolume for volume %s", pvcWithResizeRequest.QualifiedName())
		}
	}
}

func markPVCResizeInProgress(pvcWithResizeRequest *cache.PVCWithResizeRequest, kubeClient clientset.Interface) (*v1.PersistentVolumeClaim, error) {
	// Mark PVC as Resize Started
	progressCondition := v1.PersistentVolumeClaimCondition{
		Type:               v1.PersistentVolumeClaimResizing,
		Status:             v1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
	}
	conditions := []v1.PersistentVolumeClaimCondition{progressCondition}

	return util.UpdatePVCCondition(pvcWithResizeRequest.PVC, conditions, kubeClient)
}

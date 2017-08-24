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

package expand

import (
	"time"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/pkg/controller/volume/expand/cache"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

type SyncVolumeResize interface {
	Run(stopCh <-chan struct{})
}

type syncResize struct {
	loopPeriod  time.Duration
	resizeMap   cache.VolumeResizeMap
	opsExecutor operationexecutor.OperationExecutor
}

// NewSyncVolumeResize returns actual volume resize handler
func NewSyncVolumeResize(
	loopPeriod time.Duration,
	opsExecutor operationexecutor.OperationExecutor,
	resizeMap cache.VolumeResizeMap) SyncVolumeResize {
	rc := &syncResize{
		loopPeriod:  loopPeriod,
		opsExecutor: opsExecutor,
		resizeMap:   resizeMap,
	}
	return rc
}

func (rc *syncResize) Run(stopCh <-chan struct{}) {
	wait.Until(rc.Sync, rc.loopPeriod, stopCh)
}

func (rc *syncResize) Sync() {
	// Resize PVCs that require resize
	for _, pvcWithResizeRequest := range rc.resizeMap.GetPvcsWithResizeRequest() {
		uniqueVolumeKey := v1.UniqueVolumeName(pvcWithResizeRequest.UniquePvcKey())
		if rc.opsExecutor.IsOperationPending(uniqueVolumeKey, "") {
			glog.V(10).Infof("Operation for PVC %v is already pending", pvcWithResizeRequest.UniquePvcKey())
			continue
		}
		growFuncError := rc.opsExecutor.ExpandVolume(pvcWithResizeRequest, rc.resizeMap)
		if growFuncError != nil {
			glog.Errorf("Error growing pvc with %v", growFuncError)
		}
		glog.Infof("Resizing PVC %s", pvcWithResizeRequest.CurrentSize)
	}

	// For PVCs whose API objects updates failed the first time, try again
	for _, pvcWithUpdateNeeded := range rc.resizeMap.GetPvcsWithUpdateNeeded() {
		switch *pvcWithUpdateNeeded.UpdateNeeded {
		case cache.Resized:
			rc.resizeMap.MarkAsResized(pvcWithUpdateNeeded)
		case cache.ResizeFailed:
			rc.resizeMap.MarkResizeFailed(pvcWithUpdateNeeded, *pvcWithUpdateNeeded.FailedReason)
		case cache.FsResize:
			rc.resizeMap.MarkForFileSystemResize(pvcWithUpdateNeeded)
		}
	}
}

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

// Package reconciler implements interfaces that attempt to reconcile the
// desired snapshots specified in the index.
package reconciler

import (
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/controller/volume/snapshot/cache"
	"k8s.io/kubernetes/pkg/util/goroutinemap/exponentialbackoff"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/volume/util/nestedpendingoperations"
	"k8s.io/kubernetes/pkg/volume/util/operationexecutor"
)

// Reconciler runs a periodic loop to reconcile the desired state of the with
// the actual state of the world by triggering attach detach operations.
// Note: This is distinct from the Reconciler implemented by the kubelet volume
// manager. This reconciles state for the attach/detach controller. That
// reconciles state for the kubelet volume manager.
type Reconciler interface {
	// Starts running the reconciliation loop which executes periodically, checks
	// if volumes that should be attached are attached and volumes that should
	// be detached are detached. If not, it will trigger attach/detach
	// operations to rectify.
	Run(stopCh <-chan struct{})
}

// NewReconciler returns a new instance of Reconciler that waits loopPeriod
// between successive executions.
// loopPeriod is the amount of time the reconciler loop waits between
// successive executions.
func NewReconciler(
	loopPeriod time.Duration,
	actualStateOfWorld cache.ActualStateOfWorld,
	snapshotOperationExecutor operationexecutor.OperationExecutor) Reconciler {
	return &reconciler{
		loopPeriod:                loopPeriod,
		actualStateOfWorld:        actualStateOfWorld,
		snapshotOperationExecutor: snapshotOperationExecutor,
	}
}

type reconciler struct {
	loopPeriod                time.Duration
	actualStateOfWorld        cache.ActualStateOfWorld
	snapshotOperationExecutor operationexecutor.OperationExecutor
}

func (rc *reconciler) Run(stopCh <-chan struct{}) {
	wait.Until(rc.reconciliationLoopFunc(), rc.loopPeriod, stopCh)
}

func (rc *reconciler) reconciliationLoopFunc() func() {
	return func() {
		for _, volumeToSnapshot := range rc.actualStateOfWorld.GetVolumesToSnapshot() {

			//kick off snapshot in oper exec
			glog.V(5).Infof("Attempting to start snapshot for volume %q/%q", volumeToSnapshot.PersistentVolumeClaim.Namespace, volumeToSnapshot.PersistentVolumeClaim.Name)

			err := rc.snapshotOperationExecutor.CreateSnapshot(volumeToSnapshot.VolumeToSnapshot, volumeToSnapshot.SnapshotName, rc.actualStateOfWorld)
			if err == nil {
				glog.Infof("Started CreateSnapshot for volume %q/%q", volumeToSnapshot.PersistentVolumeClaim.Namespace, volumeToSnapshot.PersistentVolumeClaim.Name)
			}
			if err != nil &&
				!nestedpendingoperations.IsAlreadyExists(err) &&
				!exponentialbackoff.IsExponentialBackoff(err) {
				// Ignore nestedpendingoperations.IsAlreadyExists && exponentialbackoff.IsExponentialBackoff errors, they are expected.
				// Log all other errors.
				glog.Errorf(
					"operationExecutor.CreateSnapshot failed to start for PersistentVolumeClaim %q (spec name:%q) with err: %v",
					volumeToSnapshot.VolumeName,
					volumeToSnapshot.VolumeSpec.Name(),
					err)
			}
		}

	}
}

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

package deletionhelper

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/runtime"

	"github.com/golang/glog"
)

type HasFinalizerFunc func(runtime.Object, string) bool
type RemoveFinalizerFunc func(runtime.Object, string) (runtime.Object, error)
type ObjNameFunc func(runtime.Object) string

type DeletionHelper struct {
	hasFinalizerFunc    HasFinalizerFunc
	removeFinalizerFunc RemoveFinalizerFunc
	objNameFunc         ObjNameFunc
	updateTimeout       time.Duration
	eventRecorder       record.EventRecorder
	informer            util.FederatedInformer
	updater             util.FederatedUpdater
}

func NewDeletionHelper(
	hasFinalizerFunc HasFinalizerFunc, removeFinalizerFunc RemoveFinalizerFunc,
	objNameFunc ObjNameFunc, updateTimeout time.Duration,
	eventRecorder record.EventRecorder, informer util.FederatedInformer,
	updater util.FederatedUpdater) *DeletionHelper {
	return &DeletionHelper{
		hasFinalizerFunc:    hasFinalizerFunc,
		removeFinalizerFunc: removeFinalizerFunc,
		objNameFunc:         objNameFunc,
		updateTimeout:       updateTimeout,
		eventRecorder:       eventRecorder,
		informer:            informer,
		updater:             updater,
	}
}

// Processes the deletion of given federation object to ensure that the
// corresponding objects in underlying clusters have been taken care of.
// If the federation object has FinalizerOrphan finalizer, then it just orphans
// the cluster resources and removes the finalizer from federation object.
// Else, it deletes the corresponding cluster resources.
func (dh *DeletionHelper) ProcessDeletion(obj runtime.Object) (
	runtime.Object, error) {
	objName := dh.objNameFunc(obj)
	glog.Infof("Processing cascading deletion request for object: %s", objName)
	// If the obj has FinalizerOrphan finalizer, then we need to orphan the
	// corresponding objects in underlying clusters.
	// Just remove the finalizer in that case.
	hasOrphanFinalizer := dh.hasFinalizerFunc(obj, api.FinalizerOrphan)
	if hasOrphanFinalizer {
		return dh.removeFinalizerFunc(obj, api.FinalizerOrphan)
	}

	// Else, we need to delete the obj from all underlying clusters.
	unreadyClusters, err := dh.informer.GetUnreadyClusters()
	if err != nil {
		return nil, fmt.Errorf("failed to get a list of unready clusters: %v", err)
	}
	// TODO: Handle the case when cluster resource is watched after this is executed.
	// This can happen if a namespace is deleted before its creation had been
	// observed in all underlying clusters.
	clusterNsObjs, err := dh.informer.GetTargetStore().GetFromAllClusters(objName)
	if err != nil {
		return nil, fmt.Errorf("failed to get object %s from underlying clusters: %v", objName, err)
	}
	operations := make([]util.FederatedOperation, 0)
	for _, clusterNsObj := range clusterNsObjs {
		operations = append(operations, util.FederatedOperation{
			Type:        util.OperationTypeDelete,
			ClusterName: clusterNsObj.ClusterName,
			Obj:         clusterNsObj.Object.(runtime.Object),
		})
	}
	err = dh.updater.UpdateWithOnError(operations, dh.updateTimeout, func(op util.FederatedOperation, operror error) {
		objName := dh.objNameFunc(op.Obj)
		dh.eventRecorder.Eventf(obj, api.EventTypeNormal, "DeleteInClusterFailed",
			"Failed to delete obj %s in cluster %s: %v", objName, op.ClusterName, operror)
	})
	if err != nil {
		return nil, fmt.Errorf("failed to execute updates for obj %s: %v", objName, err)
	}

	// We have now deleted the object from all *ready* clusters.
	// But still need to wait for clusters that are not ready to ensure that
	// the object has been deleted from *all* clusters.
	if len(unreadyClusters) != 0 {
		var clusterNames []string
		for _, cluster := range unreadyClusters {
			clusterNames = append(clusterNames, cluster.Name)
		}
		return nil, fmt.Errorf("waiting for clusters %s to become ready", strings.Join(clusterNames, ","))
	}
	return obj, nil
}

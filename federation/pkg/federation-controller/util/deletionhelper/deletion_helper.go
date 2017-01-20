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

// Package to help federation controllers to delete federated resources from
// underlying clusters when the resource is deleted from federation control
// plane.
package deletionhelper

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api"
	api_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/record"

	"github.com/golang/glog"
)

const (
	// Add this finalizer to a federation resource if the resource should be
	// deleted from all underlying clusters before being deleted from
	// federation control plane.
	// This is ignored if FinalizerOrphan is also present on the resource.
	// In that case, both finalizers are removed from the resource and the
	// resource is deleted from federation control plane without affecting
	// the underlying clusters.
	FinalizerDeleteFromUnderlyingClusters string = "federation.kubernetes.io/delete-from-underlying-clusters"
)

type HasFinalizerFunc func(runtime.Object, string) bool
type RemoveFinalizerFunc func(runtime.Object, string) (runtime.Object, error)
type AddFinalizerFunc func(runtime.Object, string) (runtime.Object, error)
type ObjNameFunc func(runtime.Object) string

type DeletionHelper struct {
	hasFinalizerFunc    HasFinalizerFunc
	removeFinalizerFunc RemoveFinalizerFunc
	addFinalizerFunc    AddFinalizerFunc
	objNameFunc         ObjNameFunc
	updateTimeout       time.Duration
	eventRecorder       record.EventRecorder
	informer            util.FederatedInformer
	updater             util.FederatedUpdater
}

func NewDeletionHelper(
	hasFinalizerFunc HasFinalizerFunc, removeFinalizerFunc RemoveFinalizerFunc,
	addFinalizerFunc AddFinalizerFunc, objNameFunc ObjNameFunc,
	updateTimeout time.Duration, eventRecorder record.EventRecorder,
	informer util.FederatedInformer,
	updater util.FederatedUpdater) *DeletionHelper {
	return &DeletionHelper{
		hasFinalizerFunc:    hasFinalizerFunc,
		removeFinalizerFunc: removeFinalizerFunc,
		addFinalizerFunc:    addFinalizerFunc,
		objNameFunc:         objNameFunc,
		updateTimeout:       updateTimeout,
		eventRecorder:       eventRecorder,
		informer:            informer,
		updater:             updater,
	}
}

// Ensures that the given object has both FinalizerDeleteFromUnderlyingClusters
// and FinalizerOrphan finalizers.
// We do this so that the controller is always notified when a federation resource is deleted.
// If user deletes the resource with nil DeleteOptions or
// DeletionOptions.OrphanDependents = true then the apiserver removes the orphan finalizer
// and deletion helper does a cascading deletion.
// Otherwise, deletion helper just removes the federation resource and orphans
// the corresponding resources in underlying clusters.
// This method should be called before creating objects in underlying clusters.
func (dh *DeletionHelper) EnsureFinalizers(obj runtime.Object) (
	runtime.Object, error) {
	if !dh.hasFinalizerFunc(obj, FinalizerDeleteFromUnderlyingClusters) {
		glog.V(2).Infof("Adding finalizer %s to %s", FinalizerDeleteFromUnderlyingClusters, dh.objNameFunc(obj))
		obj, err := dh.addFinalizerFunc(obj, FinalizerDeleteFromUnderlyingClusters)
		if err != nil {
			return obj, err
		}
	}
	if !dh.hasFinalizerFunc(obj, api_v1.FinalizerOrphan) {
		glog.V(2).Infof("Adding finalizer %s to %s", api_v1.FinalizerOrphan, dh.objNameFunc(obj))
		obj, err := dh.addFinalizerFunc(obj, api_v1.FinalizerOrphan)
		if err != nil {
			return obj, err
		}
	}
	return obj, nil
}

// Deletes the resources corresponding to the given federated resource from
// all underlying clusters, unless it has the FinalizerOrphan finalizer.
// Removes FinalizerOrphan and FinalizerDeleteFromUnderlyingClusters finalizers
// when done.
// Callers are expected to keep calling this (with appropriate backoff) until
// it succeeds.
func (dh *DeletionHelper) HandleObjectInUnderlyingClusters(obj runtime.Object) (
	runtime.Object, error) {
	objName := dh.objNameFunc(obj)
	glog.V(2).Infof("Handling deletion of federated dependents for object: %s", objName)
	if !dh.hasFinalizerFunc(obj, FinalizerDeleteFromUnderlyingClusters) {
		glog.V(2).Infof("obj does not have %s finalizer. Nothing to do", FinalizerDeleteFromUnderlyingClusters)
		return obj, nil
	}
	hasOrphanFinalizer := dh.hasFinalizerFunc(obj, api_v1.FinalizerOrphan)
	if hasOrphanFinalizer {
		glog.V(2).Infof("Found finalizer orphan. Nothing to do, just remove the finalizer")
		// If the obj has FinalizerOrphan finalizer, then we need to orphan the
		// corresponding objects in underlying clusters.
		// Just remove both the finalizers in that case.
		obj, err := dh.removeFinalizerFunc(obj, FinalizerDeleteFromUnderlyingClusters)
		if err != nil {
			return obj, err
		}
		return dh.removeFinalizerFunc(obj, api_v1.FinalizerOrphan)
	}

	glog.V(2).Infof("Deleting obj %s from underlying clusters", objName)
	// Else, we need to delete the obj from all underlying clusters.
	unreadyClusters, err := dh.informer.GetUnreadyClusters()
	if err != nil {
		return nil, fmt.Errorf("failed to get a list of unready clusters: %v", err)
	}
	// TODO: Handle the case when cluster resource is watched after this is executed.
	// This can happen if a namespace is deleted before its creation had been
	// observed in all underlying clusters.
	storeKey := dh.informer.GetTargetStore().GetKeyFor(obj)
	clusterNsObjs, err := dh.informer.GetTargetStore().GetFromAllClusters(storeKey)
	glog.V(3).Infof("Found %d objects in underlying clusters", len(clusterNsObjs))
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
	if len(operations) > 0 {
		// We have deleted a bunch of resources.
		// Wait for the store to observe all the deletions.
		var clusterNames []string
		for _, op := range operations {
			clusterNames = append(clusterNames, op.ClusterName)
		}
		return nil, fmt.Errorf("waiting for object %s to be deleted from clusters: %s", objName, strings.Join(clusterNames, ", "))
	}

	// We have now deleted the object from all *ready* clusters.
	// But still need to wait for clusters that are not ready to ensure that
	// the object has been deleted from *all* clusters.
	if len(unreadyClusters) != 0 {
		var clusterNames []string
		for _, cluster := range unreadyClusters {
			clusterNames = append(clusterNames, cluster.Name)
		}
		return nil, fmt.Errorf("waiting for clusters %s to become ready to verify that obj %s has been deleted", strings.Join(clusterNames, ", "), objName)
	}

	// All done. Just remove the finalizer.
	return dh.removeFinalizerFunc(obj, FinalizerDeleteFromUnderlyingClusters)
}

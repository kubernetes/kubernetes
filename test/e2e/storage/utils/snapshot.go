/*
Copyright 2020 The Kubernetes Authors.

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

package utils

import (
	"context"
	"fmt"
	"time"

	"github.com/onsi/ginkgo/v2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	// SnapshotGroup is the snapshot CRD api group
	SnapshotGroup = "snapshot.storage.k8s.io"
	// SnapshotAPIVersion is the snapshot CRD api version
	SnapshotAPIVersion = "snapshot.storage.k8s.io/v1"
)

var (
	// SnapshotGVR is GroupVersionResource for volumesnapshots
	SnapshotGVR = schema.GroupVersionResource{Group: SnapshotGroup, Version: "v1", Resource: "volumesnapshots"}
	// SnapshotClassGVR is GroupVersionResource for volumesnapshotclasses
	SnapshotClassGVR = schema.GroupVersionResource{Group: SnapshotGroup, Version: "v1", Resource: "volumesnapshotclasses"}
	// SnapshotContentGVR is GroupVersionResource for volumesnapshotcontents
	SnapshotContentGVR = schema.GroupVersionResource{Group: SnapshotGroup, Version: "v1", Resource: "volumesnapshotcontents"}
)

// WaitForSnapshotReady waits for a VolumeSnapshot to be ready to use or until timeout occurs, whichever comes first.
func WaitForSnapshotReady(c dynamic.Interface, ns string, snapshotName string, poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for VolumeSnapshot %s to become ready", timeout, snapshotName)

	if successful := WaitUntil(poll, timeout, func() bool {
		snapshot, err := c.Resource(SnapshotGVR).Namespace(ns).Get(context.TODO(), snapshotName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Failed to get snapshot %q, retrying in %v. Error: %v", snapshotName, poll, err)
			return false
		}

		status := snapshot.Object["status"]
		if status == nil {
			framework.Logf("VolumeSnapshot %s found but is not ready.", snapshotName)
			return false
		}
		value := status.(map[string]interface{})
		if value["readyToUse"] == true {
			framework.Logf("VolumeSnapshot %s found and is ready", snapshotName)
			return true
		}

		framework.Logf("VolumeSnapshot %s found but is not ready.", snapshotName)
		return false
	}); successful {
		return nil
	}

	return fmt.Errorf("VolumeSnapshot %s is not ready within %v", snapshotName, timeout)
}

// GetSnapshotContentFromSnapshot returns the VolumeSnapshotContent object Bound to a
// given VolumeSnapshot
func GetSnapshotContentFromSnapshot(dc dynamic.Interface, snapshot *unstructured.Unstructured) *unstructured.Unstructured {
	defer ginkgo.GinkgoRecover()
	err := WaitForSnapshotReady(dc, snapshot.GetNamespace(), snapshot.GetName(), framework.Poll, framework.SnapshotCreateTimeout)
	framework.ExpectNoError(err)

	vs, err := dc.Resource(SnapshotGVR).Namespace(snapshot.GetNamespace()).Get(context.TODO(), snapshot.GetName(), metav1.GetOptions{})

	snapshotStatus := vs.Object["status"].(map[string]interface{})
	snapshotContentName := snapshotStatus["boundVolumeSnapshotContentName"].(string)
	framework.Logf("received snapshotStatus %v", snapshotStatus)
	framework.Logf("snapshotContentName %s", snapshotContentName)
	framework.ExpectNoError(err)

	vscontent, err := dc.Resource(SnapshotContentGVR).Get(context.TODO(), snapshotContentName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	return vscontent

}

// DeleteSnapshotWithoutWaiting deletes a VolumeSnapshot and return directly without waiting
func DeleteSnapshotWithoutWaiting(dc dynamic.Interface, ns string, snapshotName string) error {
	ginkgo.By("deleting the snapshot")
	err := dc.Resource(SnapshotGVR).Namespace(ns).Delete(context.TODO(), snapshotName, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

// DeleteAndWaitSnapshot deletes a VolumeSnapshot and waits for it to be deleted or until timeout occurs, whichever comes first
func DeleteAndWaitSnapshot(dc dynamic.Interface, ns string, snapshotName string, poll, timeout time.Duration) error {
	var err error
	err = DeleteSnapshotWithoutWaiting(dc, ns, snapshotName)
	if err != nil {
		return err
	}

	ginkgo.By("checking the Snapshot has been deleted")
	err = WaitForNamespacedGVRDeletion(dc, SnapshotGVR, ns, snapshotName, poll, timeout)

	return err
}

// GenerateSnapshotClassSpec constructs a new SnapshotClass instance spec
// with a unique name that is based on namespace + suffix.
func GenerateSnapshotClassSpec(
	snapshotter string,
	parameters map[string]string,
	ns string,
) *unstructured.Unstructured {
	snapshotClass := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeSnapshotClass",
			"apiVersion": SnapshotAPIVersion,
			"metadata": map[string]interface{}{
				// Name must be unique, so let's base it on namespace name and use GenerateName
				"name": names.SimpleNameGenerator.GenerateName(ns),
			},
			"driver":         snapshotter,
			"parameters":     parameters,
			"deletionPolicy": "Delete",
		},
	}

	return snapshotClass
}

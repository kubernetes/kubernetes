/*
Copyright 2023 The Kubernetes Authors.

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

package storageversion

import (
	"context"
	"fmt"
	"time"

	apiserverinternalv1alpha1 "k8s.io/api/apiserverinternal/v1alpha1"
	apiextensionshelpers "k8s.io/apiextensions-apiserver/pkg/apihelpers"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	genericstorageversion "k8s.io/apiserver/pkg/storageversion"
)

// Manager provides methods for updating StorageVersion for CRDs. It does
// goroutine management to allow CRD storage version updates running in the
// background and not blocking the caller.
type Manager interface {
	// UpdateStorageVersion updates a StorageVesrion for the given
	// CRD and returns immediately. Optionally, the caller may specify a
	// non-nil waitCh and/or a non-nil processedCh.
	// A non-nil waitCh will block the StorageVersion update until waitCh is
	// closed.
	// The manager will close the non-nil processedCh if it finished
	// processing the StorageVersion update (note that the update can either
	// succeeded or failed).
	UpdateStorageVersion(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition,
		waitCh <-chan struct{}, processedCh chan<- struct{}, errCh chan<- struct{}) error
}

// manager implements the Manager interface.
type manager struct {
	// client is the client interface that manager uses to update
	// StorageVersion objects.
	client genericstorageversion.Client
	// apiserverID is the ID of the apiserver that invokes this manager.
	apiserverID string
}

// NewManager creates a CRD StorageVersion Manager.
func NewManager(client genericstorageversion.Client, apiserverID string) Manager {
	return &manager{
		client:      client,
		apiserverID: apiserverID,
	}
}

// UpdateStorageVersion updates a StorageVesrion for the given
// CRD and returns immediately. Optionally, the caller may specify a
// non-nil waitCh and/or a non-nil processedCh.
// A non-nil waitCh will block the StorageVersion update until waitCh is
// closed.
// The manager will close the non-nil processedCh if it finished
// processing the StorageVersion update (note that the update can either
// succeeded or failed).
func (m *manager) UpdateStorageVersion(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition,
	waitCh <-chan struct{}, processedCh chan<- struct{}, errCh chan<- struct{}) error {

	if waitCh != nil {
		done := false
		for {
			select {
			case <-waitCh:
				done = true
			case <-ctx.Done():
				if errCh != nil {
					close(errCh)
				}
				return fmt.Errorf("aborted updating CRD storage version update: %v", ctx.Err())
			case <-time.After(1 * time.Minute):
				if errCh != nil {
					close(errCh)
				}
				return fmt.Errorf("timeout waiting for waitCh to close before proceeding with storageversion update for %v", crd)
			}
			if done {
				break
			}
		}
	}

	if err := m.updateCRDStorageVersion(ctx, crd); err != nil {
		utilruntime.HandleError(err)
		if errCh != nil {
			close(errCh)
		}
		return fmt.Errorf("error while updating storage version for crd %v: %v", crd, err)
	}
	// close processCh after the update is done
	if processedCh != nil {
		close(processedCh)
	}

	return nil
}

func (m *manager) updateCRDStorageVersion(ctx context.Context, crd *apiextensionsv1.CustomResourceDefinition) error {
	gr := schema.GroupResource{
		Group:    crd.Spec.Group,
		Resource: crd.Spec.Names.Plural,
	}
	storageVersion, err := apiextensionshelpers.GetCRDStorageVersion(crd)
	if err != nil {
		// This should never happened if crd is valid, which is true since we
		// only update storage version for CRDs that have been written to the
		// storage.
		return err
	}
	encodingVersion := crd.Spec.Group + "/" + storageVersion
	var servedVersions, decodableVersions []string
	for _, v := range crd.Spec.Versions {
		decodableVersions = append(decodableVersions, crd.Spec.Group+"/"+v.Name)
		if v.Served {
			servedVersions = append(servedVersions, crd.Spec.Group+"/"+v.Name)
		}
	}

	appendOwnerRefFunc := func(sv *apiserverinternalv1alpha1.StorageVersion) error {
		ref := metav1.OwnerReference{
			APIVersion: apiextensionsv1.SchemeGroupVersion.String(),
			Kind:       "CustomResourceDefinition",
			Name:       crd.Name,
			UID:        crd.UID,
		}
		for _, r := range sv.OwnerReferences {
			if r == ref {
				return nil
			}
		}
		sv.OwnerReferences = append(sv.OwnerReferences, ref)
		return nil
	}
	return genericstorageversion.UpdateStorageVersionFor(
		ctx,
		m.client,
		m.apiserverID,
		gr,
		encodingVersion,
		decodableVersions,
		servedVersions,
		appendOwnerRefFunc)
}

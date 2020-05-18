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

package meta

import (
	"context"
	"encoding/json"
	"fmt"

	"k8s.io/client-go/dynamic"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/retry"
	"k8s.io/klog/v2"
)

type objectForFinalizersPatch struct {
	ObjectMetaForFinalizersPatch `json:"metadata"`
}

// ObjectMetaForFinalizersPatch defines object meta struct for finalizers patch operation.
type ObjectMetaForFinalizersPatch struct {
	ResourceVersion string   `json:"resourceVersion"`
	Finalizers      []string `json:"finalizers"`
}

// AddFinalizer adds the finalizer to the metadata if it isn't already present and the instance is not deleted.
func AddFinalizer(ctx context.Context, client dynamic.NamespaceableResourceInterface, namespace, name, finalizerName string) error {
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		currObj, err := client.Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return nil
		}
		if err != nil {
			return err
		}

		// don't add the finalizer to a deleted instance
		if !currObj.GetDeletionTimestamp().IsZero() {
			return nil
		}

		ownerObjectWithFinalizer := meta.AddFinalizer(currObj, finalizerName).(meta.FinalizeableObject)
		if meta.FinalizersEqual(ownerObjectWithFinalizer, currObj) {
			klog.V(4).Infof("the %s finalizer is already removed from object %q -n %q", finalizerName, namespace, name)
			return nil
		}

		// remove the owner from dependent's OwnerReferences
		patch, err := json.Marshal(&objectForFinalizersPatch{
			ObjectMetaForFinalizersPatch: ObjectMetaForFinalizersPatch{
				ResourceVersion: currObj.GetResourceVersion(),
				Finalizers:      ownerObjectWithFinalizer.GetFinalizers(),
			},
		})
		if err != nil {
			return fmt.Errorf("unable to add finalizer %q -n %q due to an error serializing patch: %v", namespace, name, err)
		}
		_, err = client.Namespace(namespace).Patch(ctx, name, types.MergePatchType, patch, metav1.PatchOptions{})
		return err
	})
	return err
}

// RemoveFinalizer removes the finalizer to the metadata if it is present.
func RemoveFinalizer(ctx context.Context, client dynamic.NamespaceableResourceInterface, namespace, name, finalizerName string) error {
	err := retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		currObj, err := client.Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
		if errors.IsNotFound(err) {
			return nil
		}
		if err != nil {
			return err
		}

		ownerObjectWithFinalizer := meta.RemoveFinalizer(currObj, finalizerName).(meta.FinalizeableObject)
		if meta.FinalizersEqual(ownerObjectWithFinalizer, currObj) {
			klog.V(4).Infof("the %s finalizer is already removed from object %q -n %q", finalizerName, namespace, name)
			return nil
		}

		// remove the owner from dependent's OwnerReferences
		patch, err := json.Marshal(&objectForFinalizersPatch{
			ObjectMetaForFinalizersPatch: ObjectMetaForFinalizersPatch{
				ResourceVersion: currObj.GetResourceVersion(),
				Finalizers:      ownerObjectWithFinalizer.GetFinalizers(),
			},
		})
		if err != nil {
			return fmt.Errorf("unable to add finalizer %q -n %q due to an error serializing patch: %v", namespace, name, err)
		}
		_, err = client.Namespace(namespace).Patch(ctx, name, types.MergePatchType, patch, metav1.PatchOptions{})
		return err
	})
	return err
}

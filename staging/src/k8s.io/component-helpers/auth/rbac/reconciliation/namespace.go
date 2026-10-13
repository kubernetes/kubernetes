/*
Copyright 2018 The Kubernetes Authors.

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

package reconciliation

import (
	"context"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
)

// tryEnsureNamespace gets or creates the given namespace while ignoring forbidden errors.
// It is a best effort attempt as the user may not be able to get or create namespaces.
// This allows us to handle flows where the user can only mutate roles and role bindings.
func tryEnsureNamespace(client corev1client.NamespaceInterface, namespace string) error {
	_, getErr := client.Get(context.TODO(), namespace, metav1.GetOptions{})
	if getErr == nil {
		return nil
	}

	if fatalGetErr := utilerrors.FilterOut(getErr, apierrors.IsNotFound, apierrors.IsForbidden); fatalGetErr != nil {
		return fatalGetErr
	}

	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}
	_, createErr := client.Create(context.TODO(), ns, metav1.CreateOptions{})

	return utilerrors.FilterOut(createErr, apierrors.IsAlreadyExists, apierrors.IsForbidden)
}

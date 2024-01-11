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

package utils

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	storagev1 "k8s.io/api/storage/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/test/e2e/framework"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/kubernetes/test/utils/ktesting/kobject"
)

// LoadFromManifests is a wrapper around [kobject.LoadFromManifests].
// It opens files by name via the "testfiles" package, which means they can come from a file system
// or be built into the binary.
func LoadFromManifests(fileNames ...string) ([]interface{}, error) {
	var files []kobject.NamedReader
	for _, fileName := range fileNames {
		file, err := e2etestfiles.Open(fileName)
		if err != nil {
			return nil, err
		}
		defer file.Close()
		files = append(files, file)
	}
	return kobject.LoadFromManifests(files...)
}

// PatchItems modifies the given items in place such that each test
// gets its own instances, to avoid conflicts between different tests
// and between tests and normal deployments.
//
// This is done by:
// - creating namespaced items inside the test's namespace
// - changing the name of non-namespaced items like ClusterRole
//
// PatchItems has some limitations:
// - only some common items are supported, unknown ones trigger an error
// - only the latest stable API version for each item is supported
func PatchItems(f *framework.Framework, driverNamespace *v1.Namespace, items ...interface{}) error {
	for _, item := range items {
		// Uncomment when debugging the loading and patching of items.
		// Logf("patching original content of %T:\n%s", item, PrettyPrint(item))
		if err := patchItemRecursively(f, driverNamespace, item); err != nil {
			return err
		}
	}
	return nil
}

// CreateItems creates the items. Each of them must implement runtime.Object.
// That it uses interface{} is parameter type is historic.
//
// It returns either a cleanup function or an error, but never both.
//
// All items are deleted automatically when the test is done.
func CreateItems(ctx context.Context, f *framework.Framework, ns *v1.Namespace, items ...interface{}) error {
	ginkgo.GinkgoHelper()
	tCtx := f.TContext(ctx)
	tCtx = kobject.WithNamespace(tCtx, ns.Name)

	for _, item := range items {
		_, err := kobject.Create(tCtx, item.(runtime.Object), metav1.CreateOptions{})
		if err != nil {
			return err
		}
	}

	return nil
}

// CreateFromManifests is a combination of LoadFromManifests,
// PatchItems, patching with an optional custom function,
// and CreateItems.
func CreateFromManifests(ctx context.Context, f *framework.Framework, driverNamespace *v1.Namespace, patch func(item interface{}) error, files ...string) error {
	ginkgo.GinkgoHelper()
	items, err := LoadFromManifests(files...)
	if err != nil {
		return fmt.Errorf("CreateFromManifests: %w", err)
	}
	if err := PatchItems(f, driverNamespace, items...); err != nil {
		return err
	}
	if patch != nil {
		for _, item := range items {
			if err := patch(item); err != nil {
				return err
			}
		}
	}
	return CreateItems(ctx, f, driverNamespace, items...)
}

// patchName makes the name of some item unique by appending the
// generated unique name.
func patchName(f *framework.Framework, item *string) {
	if *item != "" {
		*item = *item + "-" + f.UniqueName
	}
}

// patchNamespace moves the item into the test's namespace.  Not
// all items can be namespaced. For those, the name also needs to be
// patched.
func patchNamespace(f *framework.Framework, driverNamespace *v1.Namespace, item *string) {
	if driverNamespace != nil {
		*item = driverNamespace.GetName()
		return
	}

	if f.Namespace != nil {
		*item = f.Namespace.GetName()
	}
}

func patchItemRecursively(f *framework.Framework, driverNamespace *v1.Namespace, item interface{}) error {
	switch item := item.(type) {
	case *rbacv1.Subject:
		patchNamespace(f, driverNamespace, &item.Namespace)
	case *rbacv1.RoleRef:
		// TODO: avoid hard-coding this special name. Perhaps add a Framework.PredefinedRoles
		// which contains all role names that are defined cluster-wide before the test starts?
		// All those names are exempt from renaming. That list could be populated by querying
		// and get extended by tests.
		if item.Name != "e2e-test-privileged-psp" {
			patchName(f, &item.Name)
		}
	case *rbacv1.ClusterRole:
		patchName(f, &item.Name)
	case *rbacv1.Role:
		patchNamespace(f, driverNamespace, &item.Namespace)
		// Roles are namespaced, but because for RoleRef above we don't
		// know whether the referenced role is a ClusterRole or Role
		// and therefore always renames, we have to do the same here.
		patchName(f, &item.Name)
	case *storagev1.StorageClass:
		patchName(f, &item.Name)
	case *storagev1.CSIDriver:
		patchName(f, &item.Name)
	case *v1.ServiceAccount:
		patchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
	case *v1.Secret:
		patchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
	case *rbacv1.ClusterRoleBinding:
		patchName(f, &item.Name)
		for i := range item.Subjects {
			if err := patchItemRecursively(f, driverNamespace, &item.Subjects[i]); err != nil {
				return fmt.Errorf("%T: %w", f, err)
			}
		}
		if err := patchItemRecursively(f, driverNamespace, &item.RoleRef); err != nil {
			return fmt.Errorf("%T: %w", f, err)
		}
	case *rbacv1.RoleBinding:
		patchNamespace(f, driverNamespace, &item.Namespace)
		for i := range item.Subjects {
			if err := patchItemRecursively(f, driverNamespace, &item.Subjects[i]); err != nil {
				return fmt.Errorf("%T: %w", f, err)
			}
		}
		if err := patchItemRecursively(f, driverNamespace, &item.RoleRef); err != nil {
			return fmt.Errorf("%T: %w", f, err)
		}
	case *v1.Service:
		patchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
	case *appsv1.StatefulSet:
		patchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
		if err := patchContainerImages(item.Spec.Template.Spec.Containers); err != nil {
			return err
		}
		if err := patchContainerImages(item.Spec.Template.Spec.InitContainers); err != nil {
			return err
		}
	case *appsv1.Deployment:
		patchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
		if err := patchContainerImages(item.Spec.Template.Spec.Containers); err != nil {
			return err
		}
		if err := patchContainerImages(item.Spec.Template.Spec.InitContainers); err != nil {
			return err
		}
	case *appsv1.DaemonSet:
		patchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
		if err := patchContainerImages(item.Spec.Template.Spec.Containers); err != nil {
			return err
		}
		if err := patchContainerImages(item.Spec.Template.Spec.InitContainers); err != nil {
			return err
		}
	case *appsv1.ReplicaSet:
		patchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
		if err := patchContainerImages(item.Spec.Template.Spec.Containers); err != nil {
			return err
		}
		if err := patchContainerImages(item.Spec.Template.Spec.InitContainers); err != nil {
			return err
		}
	case *apiextensionsv1.CustomResourceDefinition:
		// Do nothing. Patching name to all CRDs won't always be the expected behavior.
	default:
		return fmt.Errorf("missing support for patching item of type %T", item)
	}
	return nil
}

// patchContainerImages replaces the specified Container Registry with a custom
// one provided via the KUBE_TEST_REPO_LIST env variable
func patchContainerImages(containers []v1.Container) error {
	var err error
	for i, c := range containers {
		containers[i].Image, err = imageutils.ReplaceRegistryInImageURL(c.Image)
		if err != nil {
			return err
		}
	}

	return nil
}

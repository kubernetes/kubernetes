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
	"fmt"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	storagev1 "k8s.io/api/storage/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/kubernetes/test/utils/ktesting/kobject"
)

// LoadFromManifests is a wrapper around [kobject.LoadFromManifests].
// It opens files by name via the "testfiles" package, which means they can come from a file system
// or be built into the binary.
func LoadFromManifests(tCtx ktesting.TContext, fileNames ...string) []runtime.Object {
	tCtx.Helper()
	var files []kobject.NamedReader
	for _, fileName := range fileNames {
		file, err := e2etestfiles.Open(fileName)
		if err != nil {
			return nil
		}
		defer func() {
			_ = file.Close()
		}()
		files = append(files, file)
	}
	return kobject.LoadFromManifests(tCtx, nil, files...)
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
func PatchItems(tCtx ktesting.TContext, objs ...runtime.Object) {
	tCtx.Helper()
	for _, obj := range objs {
		// Uncomment when debugging the loading and patching of items.
		// Logf("patching original content of %T:\n%s", item, PrettyPrint(item))
		patchItemRecursively(tCtx, obj)
	}
}

// CreateItems creates the objects in the namespace associated with
// the context.
//
// Errors are treated as test failures.
//
// All items are deleted automatically when the test is done.
func CreateItems(tCtx ktesting.TContext, objs ...runtime.Object) {
	tCtx.Helper()

	for _, obj := range objs {
		_, err := kobject.Create(tCtx, obj, metav1.CreateOptions{})
		tCtx.ExpectNoError(err, "create object")
	}
}

// CreateFromManifests is a combination of LoadFromManifests,
// PatchItems, patching with an optional custom function,
// and CreateItems.
func CreateFromManifests(tCtx ktesting.TContext, patch func(ktesting.TContext, runtime.Object), files ...string) {
	tCtx.Helper()
	objs := LoadFromManifests(ktesting.WithStep(tCtx, "load manifests"), files...)
	PatchItems(ktesting.WithStep(tCtx, "patch items"), objs...)
	if patch != nil {
		for _, obj := range objs {
			patch(tCtx, obj)
		}
	}
	CreateItems(ktesting.WithStep(tCtx, "create items"), objs...)
}

func patchItemRecursively(tCtx ktesting.TContext, item interface{}) {
	tCtx.Helper()
	tCtx = ktesting.WithStep(tCtx, fmt.Sprintf("patching %T", item))
	namespace := tCtx.Namespace()
	switch item := item.(type) {
	case *rbacv1.Subject:
		item.Namespace = namespace
	case *rbacv1.RoleRef:
		// TODO: avoid hard-coding this special name. Perhaps add a Framework.PredefinedRoles
		// which contains all role names that are defined cluster-wide before the test starts?
		// All those names are exempt from renaming. That list could be populated by querying
		// and get extended by tests.
		if item.Name != "e2e-test-privileged-psp" {
			item.Name += "-" + namespace
		}
	case *rbacv1.ClusterRole:
		item.Name += "-" + namespace
	case *rbacv1.Role:
		item.Namespace = namespace
		// Roles are namespaced, but because for RoleRef above we don't
		// know whether the referenced role is a ClusterRole or Role
		// and therefore always renames, we have to do the same here.
		item.Name += "-" + namespace
	case *storagev1.StorageClass:
		item.Name += "-" + namespace
	case *storagev1beta1.VolumeAttributesClass:
		item.Name += "-" + namespace
	case *storagev1.CSIDriver:
		item.Name += "-" + namespace
	case *v1.ServiceAccount:
		item.ObjectMeta.Namespace = namespace
	case *v1.Secret:
		item.ObjectMeta.Namespace = namespace
	case *rbacv1.ClusterRoleBinding:
		item.Name += "-" + namespace
		for i := range item.Subjects {
			patchItemRecursively(tCtx, &item.Subjects[i])
		}
		patchItemRecursively(tCtx, &item.RoleRef)
	case *rbacv1.RoleBinding:
		item.Namespace = namespace
		for i := range item.Subjects {
			patchItemRecursively(tCtx, &item.Subjects[i])
		}
		patchItemRecursively(tCtx, &item.RoleRef)
	case *v1.Service:
		item.ObjectMeta.Namespace = namespace
	case *appsv1.StatefulSet:
		item.ObjectMeta.Namespace = namespace
		patchItemRecursively(tCtx, &item.Spec.Template.Spec)
	case *appsv1.Deployment:
		item.ObjectMeta.Namespace = namespace
		patchItemRecursively(tCtx, &item.Spec.Template.Spec)
	case *appsv1.DaemonSet:
		item.ObjectMeta.Namespace = namespace
		patchItemRecursively(tCtx, &item.Spec.Template.Spec)
	case *appsv1.ReplicaSet:
		item.ObjectMeta.Namespace = namespace
		patchItemRecursively(tCtx, &item.Spec.Template.Spec)
	case *v1.PodSpec:
		patchContainerImages(ktesting.WithStep(tCtx, "containers"), item.Containers)
		patchContainerImages(ktesting.WithStep(tCtx, "initContainers"), item.InitContainers)
	case *apiextensionsv1.CustomResourceDefinition:
		// Do nothing. Patching name to all CRDs won't always be the expected behavior.
	default:
		// WithStep provides the necessary context for how we got here.
		tCtx.Fatal("missing support for patching item")
	}
}

// patchContainerImages replaces the specified Container Registry with a custom
// one provided via the KUBE_TEST_REPO_LIST env variable
func patchContainerImages(tCtx ktesting.TContext, containers []v1.Container) {
	tCtx.Helper()
	var err error
	for i, c := range containers {
		containers[i].Image, err = imageutils.ReplaceRegistryInImageURL(c.Image)
		tCtx.ExpectNoError(err, fmt.Sprintf("patching container image %s", c.Image))
	}
}

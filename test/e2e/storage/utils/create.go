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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/onsi/ginkgo/v2"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	storagev1 "k8s.io/api/storage/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/test/e2e/framework"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

// LoadFromManifests loads .yaml or .json manifest files and returns
// all items that it finds in them. It supports all items for which
// there is a factory registered in factories and .yaml files with
// multiple items separated by "---". Files are accessed via the
// "testfiles" package, which means they can come from a file system
// or be built into the binary.
//
// LoadFromManifests has some limitations:
//   - aliases are not supported (i.e. use serviceAccountName instead of the deprecated serviceAccount,
//     https://kubernetes.io/docs/reference/kubernetes-api/workload-resources/pod-v1)
//     and silently ignored
//   - the latest stable API version for each item is used, regardless of what
//     is specified in the manifest files
func LoadFromManifests(files ...string) ([]interface{}, error) {
	var items []interface{}
	err := visitManifests(func(data []byte) error {
		// Ignore any additional fields for now, just determine what we have.
		var what What
		if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), data, &what); err != nil {
			return fmt.Errorf("decode TypeMeta: %w", err)
		}
		// Ignore empty documents.
		if what.Kind == "" {
			return nil
		}

		factory := factories[what]
		if factory == nil {
			return fmt.Errorf("item of type %+v not supported", what)
		}

		object := factory.New()
		if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), data, object); err != nil {
			return fmt.Errorf("decode %+v: %w", what, err)
		}
		items = append(items, object)
		return nil
	}, files...)

	return items, err
}

func visitManifests(cb func([]byte) error, files ...string) error {
	for _, fileName := range files {
		data, err := e2etestfiles.Read(fileName)
		if err != nil {
			framework.Failf("reading manifest file: %v", err)
		}

		// Split at the "---" separator before working on
		// individual item. Only works for .yaml.
		//
		// We need to split ourselves because we need access
		// to each original chunk of data for
		// runtime.DecodeInto. kubectl has its own
		// infrastructure for this, but that is a lot of code
		// with many dependencies.
		items := bytes.Split(data, []byte("\n---"))

		for _, item := range items {
			if err := cb(item); err != nil {
				return fmt.Errorf("%s: %w", fileName, err)
			}
		}
	}
	return nil
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

// CreateItems creates the items. Each of them must be an API object
// of a type that is registered in Factory.
//
// It returns either a cleanup function or an error, but never both.
//
// Cleaning up after a test can be triggered in two ways:
//   - the test invokes the returned cleanup function,
//     usually in an AfterEach
//   - the test suite terminates, potentially after
//     skipping the test's AfterEach (https://github.com/onsi/ginkgo/issues/222)
//
// PatchItems has the some limitations as LoadFromManifests:
// - only some common items are supported, unknown ones trigger an error
// - only the latest stable API version for each item is supported
func CreateItems(ctx context.Context, f *framework.Framework, ns *v1.Namespace, items ...interface{}) error {
	var result error
	for _, item := range items {
		// Each factory knows which item(s) it supports, so try each one.
		done := false
		description := describeItem(item)
		// Uncomment this line to get a full dump of the entire item.
		// description = fmt.Sprintf("%s:\n%s", description, PrettyPrint(item))
		framework.Logf("creating %s", description)
		for _, factory := range factories {
			destructor, err := factory.Create(ctx, f, ns, item)
			if destructor != nil {
				ginkgo.DeferCleanup(framework.IgnoreNotFound(destructor), framework.AnnotatedLocation(fmt.Sprintf("deleting %s", description)))
			}
			if err == nil {
				done = true
				break
			} else if !errors.Is(err, errorItemNotSupported) {
				result = err
				break
			}
		}
		if result == nil && !done {
			result = fmt.Errorf("item of type %T not supported", item)
			break
		}
	}

	return result
}

// CreateFromManifests is a combination of LoadFromManifests,
// PatchItems, patching with an optional custom function,
// and CreateItems.
func CreateFromManifests(ctx context.Context, f *framework.Framework, driverNamespace *v1.Namespace, patch func(item interface{}) error, files ...string) error {
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

// What is a subset of metav1.TypeMeta which (in contrast to
// metav1.TypeMeta itself) satisfies the runtime.Object interface.
type What struct {
	Kind string `json:"kind"`
}

// DeepCopy is an autogenerated deepcopy function, copying the receiver, creating a new What.
func (in *What) DeepCopy() *What {
	return &What{Kind: in.Kind}
}

// DeepCopyInto is an autogenerated deepcopy function, copying the receiver, writing into out.
func (in *What) DeepCopyInto(out *What) {
	out.Kind = in.Kind
}

// DeepCopyObject is an autogenerated deepcopy function, copying the receiver, creating a new runtime.Object.
func (in *What) DeepCopyObject() runtime.Object {
	return &What{Kind: in.Kind}
}

// GetObjectKind returns the ObjectKind schema
func (in *What) GetObjectKind() schema.ObjectKind {
	return nil
}

// ItemFactory provides support for creating one particular item.
// The type gets exported because other packages might want to
// extend the set of pre-defined factories.
type ItemFactory interface {
	// New returns a new empty item.
	New() runtime.Object

	// Create is responsible for creating the item. It returns an
	// error or a cleanup function for the created item.
	// If the item is of an unsupported type, it must return
	// an error that has errorItemNotSupported as cause.
	Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, item interface{}) (func(ctx context.Context) error, error)
}

// describeItem always returns a string that describes the item,
// usually by calling out to cache.MetaNamespaceKeyFunc which
// concatenates namespace (if set) and name. If that fails, the entire
// item gets converted to a string.
func describeItem(item interface{}) string {
	key, err := cache.MetaNamespaceKeyFunc(item)
	if err == nil && key != "" {
		return fmt.Sprintf("%T: %s", item, key)
	}
	return fmt.Sprintf("%T: %s", item, item)
}

// errorItemNotSupported is the error that Create methods
// must return or wrap when they don't support the given item.
var errorItemNotSupported = errors.New("not supported")

var factories = map[What]ItemFactory{
	{"ClusterRole"}:              &clusterRoleFactory{},
	{"ClusterRoleBinding"}:       &clusterRoleBindingFactory{},
	{"CSIDriver"}:                &csiDriverFactory{},
	{"DaemonSet"}:                &daemonSetFactory{},
	{"ReplicaSet"}:               &replicaSetFactory{},
	{"Role"}:                     &roleFactory{},
	{"RoleBinding"}:              &roleBindingFactory{},
	{"Secret"}:                   &secretFactory{},
	{"Service"}:                  &serviceFactory{},
	{"ServiceAccount"}:           &serviceAccountFactory{},
	{"StatefulSet"}:              &statefulSetFactory{},
	{"Deployment"}:               &deploymentFactory{},
	{"StorageClass"}:             &storageClassFactory{},
	{"VolumeAttributesClass"}:    &volumeAttributesClassFactory{},
	{"CustomResourceDefinition"}: &customResourceDefinitionFactory{},
}

// PatchName makes the name of some item unique by appending the
// generated unique name.
func PatchName(f *framework.Framework, item *string) {
	if *item != "" {
		*item = *item + "-" + f.UniqueName
	}
}

// PatchNamespace moves the item into the test's namespace.  Not
// all items can be namespaced. For those, the name also needs to be
// patched.
func PatchNamespace(f *framework.Framework, driverNamespace *v1.Namespace, item *string) {
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
		PatchNamespace(f, driverNamespace, &item.Namespace)
	case *rbacv1.RoleRef:
		// TODO: avoid hard-coding this special name. Perhaps add a Framework.PredefinedRoles
		// which contains all role names that are defined cluster-wide before the test starts?
		// All those names are exempt from renaming. That list could be populated by querying
		// and get extended by tests.
		if item.Name != "e2e-test-privileged-psp" {
			PatchName(f, &item.Name)
		}
	case *rbacv1.ClusterRole:
		PatchName(f, &item.Name)
	case *rbacv1.Role:
		PatchNamespace(f, driverNamespace, &item.Namespace)
		// Roles are namespaced, but because for RoleRef above we don't
		// know whether the referenced role is a ClusterRole or Role
		// and therefore always renames, we have to do the same here.
		PatchName(f, &item.Name)
	case *storagev1.StorageClass:
		PatchName(f, &item.Name)
	case *storagev1.VolumeAttributesClass:
		PatchName(f, &item.Name)
	case *storagev1.CSIDriver:
		PatchName(f, &item.Name)
	case *v1.ServiceAccount:
		PatchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
	case *v1.Secret:
		PatchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
	case *rbacv1.ClusterRoleBinding:
		PatchName(f, &item.Name)
		for i := range item.Subjects {
			if err := patchItemRecursively(f, driverNamespace, &item.Subjects[i]); err != nil {
				return fmt.Errorf("%T: %w", f, err)
			}
		}
		if err := patchItemRecursively(f, driverNamespace, &item.RoleRef); err != nil {
			return fmt.Errorf("%T: %w", f, err)
		}
	case *rbacv1.RoleBinding:
		PatchNamespace(f, driverNamespace, &item.Namespace)
		for i := range item.Subjects {
			if err := patchItemRecursively(f, driverNamespace, &item.Subjects[i]); err != nil {
				return fmt.Errorf("%T: %w", f, err)
			}
		}
		if err := patchItemRecursively(f, driverNamespace, &item.RoleRef); err != nil {
			return fmt.Errorf("%T: %w", f, err)
		}
	case *v1.Service:
		PatchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
	case *appsv1.StatefulSet:
		PatchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
		if err := patchContainerImages(item.Spec.Template.Spec.Containers); err != nil {
			return err
		}
		if err := patchContainerImages(item.Spec.Template.Spec.InitContainers); err != nil {
			return err
		}
	case *appsv1.Deployment:
		PatchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
		if err := patchContainerImages(item.Spec.Template.Spec.Containers); err != nil {
			return err
		}
		if err := patchContainerImages(item.Spec.Template.Spec.InitContainers); err != nil {
			return err
		}
	case *appsv1.DaemonSet:
		PatchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
		if err := patchContainerImages(item.Spec.Template.Spec.Containers); err != nil {
			return err
		}
		if err := patchContainerImages(item.Spec.Template.Spec.InitContainers); err != nil {
			return err
		}
	case *appsv1.ReplicaSet:
		PatchNamespace(f, driverNamespace, &item.ObjectMeta.Namespace)
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

// The individual factories all follow the same template, but with
// enough differences in types and functions that copy-and-paste
// looked like the least dirty approach. Perhaps one day Go will have
// generics.

type serviceAccountFactory struct{}

func (f *serviceAccountFactory) New() runtime.Object {
	return &v1.ServiceAccount{}
}

func (*serviceAccountFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*v1.ServiceAccount)
	if !ok {
		return nil, errorItemNotSupported
	}
	client := f.ClientSet.CoreV1().ServiceAccounts(ns.Name)
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create ServiceAccount: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type clusterRoleFactory struct{}

func (f *clusterRoleFactory) New() runtime.Object {
	return &rbacv1.ClusterRole{}
}

func (*clusterRoleFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*rbacv1.ClusterRole)
	if !ok {
		return nil, errorItemNotSupported
	}

	framework.Logf("Define cluster role %v", item.GetName())
	client := f.ClientSet.RbacV1().ClusterRoles()
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create ClusterRole: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type clusterRoleBindingFactory struct{}

func (f *clusterRoleBindingFactory) New() runtime.Object {
	return &rbacv1.ClusterRoleBinding{}
}

func (*clusterRoleBindingFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*rbacv1.ClusterRoleBinding)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := f.ClientSet.RbacV1().ClusterRoleBindings()
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create ClusterRoleBinding: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type roleFactory struct{}

func (f *roleFactory) New() runtime.Object {
	return &rbacv1.Role{}
}

func (*roleFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*rbacv1.Role)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := f.ClientSet.RbacV1().Roles(ns.Name)
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create Role: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type roleBindingFactory struct{}

func (f *roleBindingFactory) New() runtime.Object {
	return &rbacv1.RoleBinding{}
}

func (*roleBindingFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*rbacv1.RoleBinding)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := f.ClientSet.RbacV1().RoleBindings(ns.Name)
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create RoleBinding: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type serviceFactory struct{}

func (f *serviceFactory) New() runtime.Object {
	return &v1.Service{}
}

func (*serviceFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*v1.Service)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := f.ClientSet.CoreV1().Services(ns.Name)
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create Service: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type statefulSetFactory struct{}

func (f *statefulSetFactory) New() runtime.Object {
	return &appsv1.StatefulSet{}
}

func (*statefulSetFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*appsv1.StatefulSet)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := f.ClientSet.AppsV1().StatefulSets(ns.Name)
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create StatefulSet: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type deploymentFactory struct{}

func (f *deploymentFactory) New() runtime.Object {
	return &appsv1.Deployment{}
}

func (*deploymentFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*appsv1.Deployment)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := f.ClientSet.AppsV1().Deployments(ns.Name)
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create Deployment: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type daemonSetFactory struct{}

func (f *daemonSetFactory) New() runtime.Object {
	return &appsv1.DaemonSet{}
}

func (*daemonSetFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*appsv1.DaemonSet)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := f.ClientSet.AppsV1().DaemonSets(ns.Name)
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create DaemonSet: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type replicaSetFactory struct{}

func (f *replicaSetFactory) New() runtime.Object {
	return &appsv1.ReplicaSet{}
}

func (*replicaSetFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*appsv1.ReplicaSet)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := f.ClientSet.AppsV1().ReplicaSets(ns.Name)
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create ReplicaSet: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type storageClassFactory struct{}

func (f *storageClassFactory) New() runtime.Object {
	return &storagev1.StorageClass{}
}

func (*storageClassFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*storagev1.StorageClass)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := f.ClientSet.StorageV1().StorageClasses()
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create StorageClass: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type volumeAttributesClassFactory struct{}

func (f *volumeAttributesClassFactory) New() runtime.Object {
	return &storagev1.VolumeAttributesClass{}
}

func (*volumeAttributesClassFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*storagev1.VolumeAttributesClass)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := f.ClientSet.StorageV1().VolumeAttributesClasses()
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create VolumeAttributesClass: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type csiDriverFactory struct{}

func (f *csiDriverFactory) New() runtime.Object {
	return &storagev1.CSIDriver{}
}

func (*csiDriverFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*storagev1.CSIDriver)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := f.ClientSet.StorageV1().CSIDrivers()
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create CSIDriver: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type secretFactory struct{}

func (f *secretFactory) New() runtime.Object {
	return &v1.Secret{}
}

func (*secretFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*v1.Secret)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := f.ClientSet.CoreV1().Secrets(ns.Name)
	if _, err := client.Create(ctx, item, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create Secret: %w", err)
	}
	return func(ctx context.Context) error {
		return client.Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

type customResourceDefinitionFactory struct{}

func (f *customResourceDefinitionFactory) New() runtime.Object {
	return &apiextensionsv1.CustomResourceDefinition{}
}

func (*customResourceDefinitionFactory) Create(ctx context.Context, f *framework.Framework, ns *v1.Namespace, i interface{}) (func(ctx context.Context) error, error) {
	var err error
	unstructCRD := &unstructured.Unstructured{}
	gvr := schema.GroupVersionResource{Group: "apiextensions.k8s.io", Version: "v1", Resource: "customresourcedefinitions"}

	item, ok := i.(*apiextensionsv1.CustomResourceDefinition)
	if !ok {
		return nil, errorItemNotSupported
	}

	unstructCRD.Object, err = runtime.DefaultUnstructuredConverter.ToUnstructured(i)
	if err != nil {
		return nil, err
	}

	if _, err = f.DynamicClient.Resource(gvr).Create(ctx, unstructCRD, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create CustomResourceDefinition: %w", err)
	}
	return func(ctx context.Context) error {
		return f.DynamicClient.Resource(gvr).Delete(ctx, item.GetName(), metav1.DeleteOptions{})
	}, nil
}

// PrettyPrint returns a human-readable representation of an item.
func PrettyPrint(item interface{}) string {
	data, err := json.MarshalIndent(item, "", "  ")
	if err == nil {
		return string(data)
	}
	return fmt.Sprintf("%+v", item)
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

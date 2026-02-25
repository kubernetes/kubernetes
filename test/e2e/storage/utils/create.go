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

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	storagev1 "k8s.io/api/storage/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/test/e2e/framework"
	e2etestfiles "k8s.io/kubernetes/test/e2e/framework/testfiles"
	imageutils "k8s.io/kubernetes/test/utils/image"
	"k8s.io/kubernetes/test/utils/ktesting"
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
	tCtx := f.TContext(context.Background())
	if driverNamespace != nil {
		tCtx = tCtx.WithNamespace(driverNamespace.Name)
	}
	return PatchItemsTCtx(tCtx, items...)
}

// PatchItemsTCtx is a variant of PatchItems where all parameters, including
// the namespace, are passed through a TContext.
func PatchItemsTCtx(tCtx ktesting.TContext, items ...interface{}) error {
	for _, item := range items {
		// Uncomment when debugging the loading and patching of items.
		// Logf("patching original content of %T:\n%s", item, PrettyPrint(item))
		if err := patchItemRecursively(tCtx, item); err != nil {
			return err
		}
	}
	return nil
}

// createItems creates the items. Each of them must be an API object
// of a type that is registered in Factory.
//
// Object get deleted automatically during test cleanup.
func createItems(tCtx ktesting.TContext, items ...interface{}) error {
	var result error
	for _, item := range items {
		// Each factory knows which item(s) it supports, so try each one.
		done := false
		description := describeItem(item)
		// Uncomment this line to get a full dump of the entire item.
		// description = fmt.Sprintf("%s:\n%s", description, PrettyPrint(item))
		tCtx.Logf("creating %s", description)
		for _, factory := range factories {
			destructor, err := factory.Create(tCtx, item)
			if destructor != nil {
				tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
					err := destructor(tCtx)
					if apierrors.IsNotFound(err) {
						return
					}
					tCtx.ExpectNoError(err, fmt.Sprintf("deleting %s", description))
				})
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
// and creating the resulting items.
//
// Objects get deleted automatically during test cleanup.
func CreateFromManifests(ctx context.Context, f *framework.Framework, driverNamespace *v1.Namespace, patch func(item interface{}) error, files ...string) error {
	tCtx := f.TContext(ctx)
	if driverNamespace != nil {
		tCtx = tCtx.WithNamespace(driverNamespace.Name)
	}
	return CreateFromManifestsTCtx(tCtx, patch, files...)
}

// CreateFromManifestsTCtx is a variant of CreateFromManifests where all parameters, including
// the driver namespace, are passed through a TContext. It is therefore usable from Go unit tests.
func CreateFromManifestsTCtx(tCtx ktesting.TContext, patch func(item interface{}) error, files ...string) error {
	items, err := LoadFromManifests(files...)
	if err != nil {
		return fmt.Errorf("CreateFromManifests: %w", err)
	}
	if err := PatchItemsTCtx(tCtx, items...); err != nil {
		return err
	}
	if patch != nil {
		for _, item := range items {
			if err := patch(item); err != nil {
				return err
			}
		}
	}
	return createItems(tCtx, items...)
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
	Create(tCtx ktesting.TContext, item interface{}) (func(ctx context.Context) error, error)
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

// patchName makes the name of some item unique by appending the
// generated unique name.
func patchName(uniqueName string, item *string) {
	if *item != "" {
		*item = *item + "-" + uniqueName
	}
}

// patchNamespace moves the item into the test's namespace.  Not
// all items can be namespaced. For those, the name also needs to be
// patched.
func patchNamespace(tCtx ktesting.TContext, item *string) {
	namespace := tCtx.Namespace()
	if namespace != "" {
		*item = namespace
	}
}

func patchItemRecursively(tCtx ktesting.TContext, item interface{}) error {
	uniqueName := tCtx.Namespace()
	switch item := item.(type) {
	case *rbacv1.Subject:
		patchNamespace(tCtx, &item.Namespace)
	case *rbacv1.RoleRef:
		// TODO: avoid hard-coding this special name. Perhaps add a Framework.PredefinedRoles
		// which contains all role names that are defined cluster-wide before the test starts?
		// All those names are exempt from renaming. That list could be populated by querying
		// and get extended by tests.
		if item.Name != "e2e-test-privileged-psp" {
			patchName(uniqueName, &item.Name)
		}
	case *rbacv1.ClusterRole:
		patchName(uniqueName, &item.Name)
	case *rbacv1.Role:
		patchNamespace(tCtx, &item.Namespace)
		// Roles are namespaced, but because for RoleRef above we don't
		// know whether the referenced role is a ClusterRole or Role
		// and therefore always renames, we have to do the same here.
		patchName(uniqueName, &item.Name)
	case *storagev1.StorageClass:
		patchName(uniqueName, &item.Name)
	case *storagev1.VolumeAttributesClass:
		patchName(uniqueName, &item.Name)
	case *storagev1.CSIDriver:
		patchName(uniqueName, &item.Name)
	case *v1.ServiceAccount:
		patchNamespace(tCtx, &item.ObjectMeta.Namespace)
	case *v1.Secret:
		patchNamespace(tCtx, &item.ObjectMeta.Namespace)
	case *rbacv1.ClusterRoleBinding:
		patchName(uniqueName, &item.Name)
		for i := range item.Subjects {
			if err := patchItemRecursively(tCtx, &item.Subjects[i]); err != nil {
				return fmt.Errorf("%T: %w", &item.Subjects[i], err)
			}
		}
		if err := patchItemRecursively(tCtx, &item.RoleRef); err != nil {
			return fmt.Errorf("%T: %w", &item.RoleRef, err)
		}
	case *rbacv1.RoleBinding:
		patchNamespace(tCtx, &item.Namespace)
		for i := range item.Subjects {
			if err := patchItemRecursively(tCtx, &item.Subjects[i]); err != nil {
				return fmt.Errorf("%T: %w", &item.Subjects[i], err)
			}
		}
		if err := patchItemRecursively(tCtx, &item.RoleRef); err != nil {
			return fmt.Errorf("%T: %w", &item.RoleRef, err)
		}
	case *v1.Service:
		patchNamespace(tCtx, &item.ObjectMeta.Namespace)
	case *appsv1.StatefulSet:
		patchNamespace(tCtx, &item.ObjectMeta.Namespace)
		if err := patchContainerImages(item.Spec.Template.Spec.Containers); err != nil {
			return err
		}
		if err := patchContainerImages(item.Spec.Template.Spec.InitContainers); err != nil {
			return err
		}
	case *appsv1.Deployment:
		patchNamespace(tCtx, &item.ObjectMeta.Namespace)
		if err := patchContainerImages(item.Spec.Template.Spec.Containers); err != nil {
			return err
		}
		if err := patchContainerImages(item.Spec.Template.Spec.InitContainers); err != nil {
			return err
		}
	case *appsv1.DaemonSet:
		patchNamespace(tCtx, &item.ObjectMeta.Namespace)
		if err := patchContainerImages(item.Spec.Template.Spec.Containers); err != nil {
			return err
		}
		if err := patchContainerImages(item.Spec.Template.Spec.InitContainers); err != nil {
			return err
		}
	case *appsv1.ReplicaSet:
		patchNamespace(tCtx, &item.ObjectMeta.Namespace)
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

func (*serviceAccountFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*v1.ServiceAccount)
	if !ok {
		return nil, errorItemNotSupported
	}
	client := tCtx.Client().CoreV1().ServiceAccounts(tCtx.Namespace())
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*clusterRoleFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*rbacv1.ClusterRole)
	if !ok {
		return nil, errorItemNotSupported
	}

	tCtx.Logf("define cluster role %v", item.GetName())
	client := tCtx.Client().RbacV1().ClusterRoles()
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*clusterRoleBindingFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*rbacv1.ClusterRoleBinding)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := tCtx.Client().RbacV1().ClusterRoleBindings()
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*roleFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*rbacv1.Role)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := tCtx.Client().RbacV1().Roles(tCtx.Namespace())
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*roleBindingFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*rbacv1.RoleBinding)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := tCtx.Client().RbacV1().RoleBindings(tCtx.Namespace())
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*serviceFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*v1.Service)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := tCtx.Client().CoreV1().Services(tCtx.Namespace())
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*statefulSetFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*appsv1.StatefulSet)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := tCtx.Client().AppsV1().StatefulSets(tCtx.Namespace())
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*deploymentFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*appsv1.Deployment)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := tCtx.Client().AppsV1().Deployments(tCtx.Namespace())
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*daemonSetFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*appsv1.DaemonSet)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := tCtx.Client().AppsV1().DaemonSets(tCtx.Namespace())
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*replicaSetFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*appsv1.ReplicaSet)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := tCtx.Client().AppsV1().ReplicaSets(tCtx.Namespace())
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*storageClassFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*storagev1.StorageClass)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := tCtx.Client().StorageV1().StorageClasses()
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*volumeAttributesClassFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*storagev1.VolumeAttributesClass)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := tCtx.Client().StorageV1().VolumeAttributesClasses()
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*csiDriverFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*storagev1.CSIDriver)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := tCtx.Client().StorageV1().CSIDrivers()
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*secretFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
	item, ok := i.(*v1.Secret)
	if !ok {
		return nil, errorItemNotSupported
	}

	client := tCtx.Client().CoreV1().Secrets(tCtx.Namespace())
	if _, err := client.Create(tCtx, item, metav1.CreateOptions{}); err != nil {
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

func (*customResourceDefinitionFactory) Create(tCtx ktesting.TContext, i interface{}) (func(ctx context.Context) error, error) {
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

	if _, err = tCtx.Dynamic().Resource(gvr).Create(tCtx, unstructCRD, metav1.CreateOptions{}); err != nil {
		return nil, fmt.Errorf("create CustomResourceDefinition: %w", err)
	}
	return func(ctx context.Context) error {
		return tCtx.Dynamic().Resource(gvr).Delete(ctx, item.GetName(), metav1.DeleteOptions{})
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

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

package framework

import (
	"bytes"
	"encoding/json"
	"fmt"

	"github.com/pkg/errors"

	apps "k8s.io/api/apps/v1"
	"k8s.io/api/core/v1"
	rbac "k8s.io/api/rbac/v1"
	storage "k8s.io/api/storage/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/test/e2e/framework/testfiles"
)

// LoadFromManifests loads .yaml or .json manifest files and returns
// all items that it finds in them. It supports all items for which
// there is a factory registered in Factories and .yaml files with
// multiple items separated by "---". Files are accessed via the
// "testfiles" package, which means they can come from a file system
// or be built into the binary.
//
// LoadFromManifests has some limitations:
// - aliases are not supported (i.e. use serviceAccountName instead of the deprecated serviceAccount,
//   https://kubernetes.io/docs/reference/generated/kubernetes-api/v1.11/#podspec-v1-core)
//   and silently ignored
// - the latest stable API version for each item is used, regardless of what
//   is specified in the manifest files
func (f *Framework) LoadFromManifests(files ...string) ([]interface{}, error) {
	var items []interface{}
	err := visitManifests(func(data []byte) error {
		// Ignore any additional fields for now, just determine what we have.
		var what What
		if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), data, &what); err != nil {
			return errors.Wrap(err, "decode TypeMeta")
		}

		factory := Factories[what]
		if factory == nil {
			return errors.Errorf("item of type %+v not supported", what)
		}

		object := factory.New()
		if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), data, object); err != nil {
			return errors.Wrapf(err, "decode %+v", what)
		}
		items = append(items, object)
		return nil
	}, files...)

	return items, err
}

func visitManifests(cb func([]byte) error, files ...string) error {
	for _, fileName := range files {
		data, err := testfiles.Read(fileName)
		if err != nil {
			Failf("reading manifest file: %v", err)
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
				return errors.Wrap(err, fileName)
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
func (f *Framework) PatchItems(items ...interface{}) error {
	for _, item := range items {
		// Uncomment when debugging the loading and patching of items.
		// Logf("patching original content of %T:\n%s", item, PrettyPrint(item))
		if err := f.patchItemRecursively(item); err != nil {
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
// - the test invokes the returned cleanup function,
//   usually in an AfterEach
// - the test suite terminates, potentially after
//   skipping the test's AfterEach (https://github.com/onsi/ginkgo/issues/222)
//
// PatchItems has the some limitations as LoadFromManifests:
// - only some common items are supported, unknown ones trigger an error
// - only the latest stable API version for each item is supported
func (f *Framework) CreateItems(items ...interface{}) (func(), error) {
	var destructors []func() error
	var cleanupHandle CleanupActionHandle
	cleanup := func() {
		if cleanupHandle == nil {
			// Already done.
			return
		}
		RemoveCleanupAction(cleanupHandle)

		// TODO (?): use same logic as framework.go for determining
		// whether we are expected to clean up? This would change the
		// meaning of the -delete-namespace and -delete-namespace-on-failure
		// command line flags, because they would also start to apply
		// to non-namespaced items.
		for _, destructor := range destructors {
			if err := destructor(); err != nil && !apierrs.IsNotFound(err) {
				Logf("deleting failed: %s", err)
			}
		}
	}
	cleanupHandle = AddCleanupAction(cleanup)

	var result error
	for _, item := range items {
		// Each factory knows which item(s) it supports, so try each one.
		done := false
		description := DescribeItem(item)
		// Uncomment this line to get a full dump of the entire item.
		// description = fmt.Sprintf("%s:\n%s", description, PrettyPrint(item))
		Logf("creating %s", description)
		for _, factory := range Factories {
			destructor, err := factory.Create(f, item)
			if destructor != nil {
				destructors = append(destructors, func() error {
					Logf("deleting %s", description)
					return destructor()
				})
			}
			if err == nil {
				done = true
				break
			} else if errors.Cause(err) != ItemNotSupported {
				result = err
				break
			}
		}
		if result == nil && !done {
			result = errors.Errorf("item of type %T not supported", item)
			break
		}
	}

	if result != nil {
		cleanup()
		return nil, result
	}

	return cleanup, nil
}

// CreateFromManifests is a combination of LoadFromManifests,
// PatchItems, patching with an optional custom function,
// and CreateItems.
func (f *Framework) CreateFromManifests(patch func(item interface{}) error, files ...string) (func(), error) {
	items, err := f.LoadFromManifests(files...)
	if err != nil {
		return nil, errors.Wrap(err, "CreateFromManifests")
	}
	if err := f.PatchItems(items...); err != nil {
		return nil, err
	}
	if patch != nil {
		for _, item := range items {
			if err := patch(item); err != nil {
				return nil, err
			}
		}
	}
	return f.CreateItems(items...)
}

// What is a subset of metav1.TypeMeta which (in contrast to
// metav1.TypeMeta itself) satisfies the runtime.Object interface.
type What struct {
	Kind string `json:"kind"`
}

func (in *What) DeepCopy() *What {
	return &What{Kind: in.Kind}
}

func (in *What) DeepCopyInto(out *What) {
	out.Kind = in.Kind
}

func (in *What) DeepCopyObject() runtime.Object {
	return &What{Kind: in.Kind}
}

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
	// an error that has ItemNotSupported as cause.
	Create(f *Framework, item interface{}) (func() error, error)
}

// DescribeItem always returns a string that describes the item,
// usually by calling out to cache.MetaNamespaceKeyFunc which
// concatenates namespace (if set) and name. If that fails, the entire
// item gets converted to a string.
func DescribeItem(item interface{}) string {
	key, err := cache.MetaNamespaceKeyFunc(item)
	if err == nil && key != "" {
		return fmt.Sprintf("%T: %s", item, key)
	}
	return fmt.Sprintf("%T: %s", item, item)
}

// ItemNotSupported is the error that Create methods
// must return or wrap when they don't support the given item.
var ItemNotSupported = errors.New("not supported")

var Factories = map[What]ItemFactory{
	{"ClusterRole"}:        &clusterRoleFactory{},
	{"ClusterRoleBinding"}: &clusterRoleBindingFactory{},
	{"DaemonSet"}:          &daemonSetFactory{},
	{"Role"}:               &roleFactory{},
	{"RoleBinding"}:        &roleBindingFactory{},
	{"Secret"}:             &secretFactory{},
	{"Service"}:            &serviceFactory{},
	{"ServiceAccount"}:     &serviceAccountFactory{},
	{"StatefulSet"}:        &statefulSetFactory{},
	{"StorageClass"}:       &storageClassFactory{},
}

// PatchName makes the name of some item unique by appending the
// generated unique name.
func (f *Framework) PatchName(item *string) {
	if *item != "" {
		*item = *item + "-" + f.UniqueName
	}
}

// PatchNamespace moves the item into the test's namespace.  Not
// all items can be namespaced. For those, the name also needs to be
// patched.
func (f *Framework) PatchNamespace(item *string) {
	if f.Namespace != nil {
		*item = f.Namespace.GetName()
	}
}

func (f *Framework) patchItemRecursively(item interface{}) error {
	switch item := item.(type) {
	case *rbac.Subject:
		f.PatchNamespace(&item.Namespace)
	case *rbac.RoleRef:
		// TODO: avoid hard-coding this special name. Perhaps add a Framework.PredefinedRoles
		// which contains all role names that are defined cluster-wide before the test starts?
		// All those names are excempt from renaming. That list could be populated by querying
		// and get extended by tests.
		if item.Name != "e2e-test-privileged-psp" {
			f.PatchName(&item.Name)
		}
	case *rbac.ClusterRole:
		f.PatchName(&item.Name)
	case *rbac.Role:
		f.PatchNamespace(&item.Namespace)
		// Roles are namespaced, but because for RoleRef above we don't
		// know whether the referenced role is a ClusterRole or Role
		// and therefore always renames, we have to do the same here.
		f.PatchName(&item.Name)
	case *storage.StorageClass:
		f.PatchName(&item.Name)
	case *v1.ServiceAccount:
		f.PatchNamespace(&item.ObjectMeta.Namespace)
	case *v1.Secret:
		f.PatchNamespace(&item.ObjectMeta.Namespace)
	case *rbac.ClusterRoleBinding:
		f.PatchName(&item.Name)
		for i := range item.Subjects {
			if err := f.patchItemRecursively(&item.Subjects[i]); err != nil {
				return errors.Wrapf(err, "%T", f)
			}
		}
		if err := f.patchItemRecursively(&item.RoleRef); err != nil {
			return errors.Wrapf(err, "%T", f)
		}
	case *rbac.RoleBinding:
		f.PatchNamespace(&item.Namespace)
		for i := range item.Subjects {
			if err := f.patchItemRecursively(&item.Subjects[i]); err != nil {
				return errors.Wrapf(err, "%T", f)
			}
		}
		if err := f.patchItemRecursively(&item.RoleRef); err != nil {
			return errors.Wrapf(err, "%T", f)
		}
	case *v1.Service:
		f.PatchNamespace(&item.ObjectMeta.Namespace)
	case *apps.StatefulSet:
		f.PatchNamespace(&item.ObjectMeta.Namespace)
	case *apps.DaemonSet:
		f.PatchNamespace(&item.ObjectMeta.Namespace)
	default:
		return errors.Errorf("missing support for patching item of type %T", item)
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

func (*serviceAccountFactory) Create(f *Framework, i interface{}) (func() error, error) {
	item, ok := i.(*v1.ServiceAccount)
	if !ok {
		return nil, ItemNotSupported
	}
	client := f.ClientSet.CoreV1().ServiceAccounts(f.Namespace.GetName())
	if _, err := client.Create(item); err != nil {
		return nil, errors.Wrap(err, "create ServiceAccount")
	}
	return func() error {
		return client.Delete(item.GetName(), &metav1.DeleteOptions{})
	}, nil
}

type clusterRoleFactory struct{}

func (f *clusterRoleFactory) New() runtime.Object {
	return &rbac.ClusterRole{}
}

func (*clusterRoleFactory) Create(f *Framework, i interface{}) (func() error, error) {
	item, ok := i.(*rbac.ClusterRole)
	if !ok {
		return nil, ItemNotSupported
	}

	// Impersonation is required for Kubernetes < 1.12, see
	// https://github.com/kubernetes/kubernetes/issues/62237#issuecomment-429315111
	//
	// This code is kept even for more recent Kubernetes, because users of
	// the framework outside of Kubernetes might run against an older version
	// of Kubernetes. It will be deprecated eventually.
	//
	// TODO: is this only needed for a ClusterRole or also for other non-namespaced
	// items?
	Logf("Creating an impersonating superuser kubernetes clientset to define cluster role")
	rc, err := LoadConfig()
	ExpectNoError(err)
	rc.Impersonate = restclient.ImpersonationConfig{
		UserName: "superuser",
		Groups:   []string{"system:masters"},
	}
	superuserClientset, err := clientset.NewForConfig(rc)
	ExpectNoError(err, "create superuser clientset")

	client := superuserClientset.RbacV1().ClusterRoles()
	if _, err = client.Create(item); err != nil {
		return nil, errors.Wrap(err, "create ClusterRole")
	}
	return func() error {
		return client.Delete(item.GetName(), &metav1.DeleteOptions{})
	}, nil
}

type clusterRoleBindingFactory struct{}

func (f *clusterRoleBindingFactory) New() runtime.Object {
	return &rbac.ClusterRoleBinding{}
}

func (*clusterRoleBindingFactory) Create(f *Framework, i interface{}) (func() error, error) {
	item, ok := i.(*rbac.ClusterRoleBinding)
	if !ok {
		return nil, ItemNotSupported
	}

	client := f.ClientSet.RbacV1().ClusterRoleBindings()
	if _, err := client.Create(item); err != nil {
		return nil, errors.Wrap(err, "create ClusterRoleBinding")
	}
	return func() error {
		return client.Delete(item.GetName(), &metav1.DeleteOptions{})
	}, nil
}

type roleFactory struct{}

func (f *roleFactory) New() runtime.Object {
	return &rbac.Role{}
}

func (*roleFactory) Create(f *Framework, i interface{}) (func() error, error) {
	item, ok := i.(*rbac.Role)
	if !ok {
		return nil, ItemNotSupported
	}

	client := f.ClientSet.RbacV1().Roles(f.Namespace.GetName())
	if _, err := client.Create(item); err != nil {
		return nil, errors.Wrap(err, "create Role")
	}
	return func() error {
		return client.Delete(item.GetName(), &metav1.DeleteOptions{})
	}, nil
}

type roleBindingFactory struct{}

func (f *roleBindingFactory) New() runtime.Object {
	return &rbac.RoleBinding{}
}

func (*roleBindingFactory) Create(f *Framework, i interface{}) (func() error, error) {
	item, ok := i.(*rbac.RoleBinding)
	if !ok {
		return nil, ItemNotSupported
	}

	client := f.ClientSet.RbacV1().RoleBindings(f.Namespace.GetName())
	if _, err := client.Create(item); err != nil {
		return nil, errors.Wrap(err, "create RoleBinding")
	}
	return func() error {
		return client.Delete(item.GetName(), &metav1.DeleteOptions{})
	}, nil
}

type serviceFactory struct{}

func (f *serviceFactory) New() runtime.Object {
	return &v1.Service{}
}

func (*serviceFactory) Create(f *Framework, i interface{}) (func() error, error) {
	item, ok := i.(*v1.Service)
	if !ok {
		return nil, ItemNotSupported
	}

	client := f.ClientSet.CoreV1().Services(f.Namespace.GetName())
	if _, err := client.Create(item); err != nil {
		return nil, errors.Wrap(err, "create Service")
	}
	return func() error {
		return client.Delete(item.GetName(), &metav1.DeleteOptions{})
	}, nil
}

type statefulSetFactory struct{}

func (f *statefulSetFactory) New() runtime.Object {
	return &apps.StatefulSet{}
}

func (*statefulSetFactory) Create(f *Framework, i interface{}) (func() error, error) {
	item, ok := i.(*apps.StatefulSet)
	if !ok {
		return nil, ItemNotSupported
	}

	client := f.ClientSet.AppsV1().StatefulSets(f.Namespace.GetName())
	if _, err := client.Create(item); err != nil {
		return nil, errors.Wrap(err, "create StatefulSet")
	}
	return func() error {
		return client.Delete(item.GetName(), &metav1.DeleteOptions{})
	}, nil
}

type daemonSetFactory struct{}

func (f *daemonSetFactory) New() runtime.Object {
	return &apps.DaemonSet{}
}

func (*daemonSetFactory) Create(f *Framework, i interface{}) (func() error, error) {
	item, ok := i.(*apps.DaemonSet)
	if !ok {
		return nil, ItemNotSupported
	}

	client := f.ClientSet.AppsV1().DaemonSets(f.Namespace.GetName())
	if _, err := client.Create(item); err != nil {
		return nil, errors.Wrap(err, "create DaemonSet")
	}
	return func() error {
		return client.Delete(item.GetName(), &metav1.DeleteOptions{})
	}, nil
}

type storageClassFactory struct{}

func (f *storageClassFactory) New() runtime.Object {
	return &storage.StorageClass{}
}

func (*storageClassFactory) Create(f *Framework, i interface{}) (func() error, error) {
	item, ok := i.(*storage.StorageClass)
	if !ok {
		return nil, ItemNotSupported
	}

	client := f.ClientSet.StorageV1().StorageClasses()
	if _, err := client.Create(item); err != nil {
		return nil, errors.Wrap(err, "create StorageClass")
	}
	return func() error {
		return client.Delete(item.GetName(), &metav1.DeleteOptions{})
	}, nil
}

type secretFactory struct{}

func (f *secretFactory) New() runtime.Object {
	return &v1.Secret{}
}

func (*secretFactory) Create(f *Framework, i interface{}) (func() error, error) {
	item, ok := i.(*v1.Secret)
	if !ok {
		return nil, ItemNotSupported
	}

	client := f.ClientSet.CoreV1().Secrets(f.Namespace.GetName())
	if _, err := client.Create(item); err != nil {
		return nil, errors.Wrap(err, "create Secret")
	}
	return func() error {
		return client.Delete(item.GetName(), &metav1.DeleteOptions{})
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

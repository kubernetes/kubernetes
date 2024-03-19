/*
Copyright 2024 The Kubernetes Authors.

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

package testing

import (
	"context"
	"fmt"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/apitesting"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/cacher"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	"k8s.io/utils/clock"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
}

func TestWatchSemanticInitialEventsExtended(t *testing.T) {
	store, terminate := testSetupWithEtcdAndCreateWrapper(t)
	t.Cleanup(terminate)
	storagetesting.RunWatchSemanticInitialEventsExtended(context.TODO(), t, store)
}

// GetPodAttrs returns labels and fields of a given object for filtering purposes.
func GetPodAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	pod, ok := obj.(*example.Pod)
	if !ok {
		return nil, nil, fmt.Errorf("not a pod")
	}
	return labels.Set(pod.ObjectMeta.Labels), PodToSelectableFields(pod), nil
}

// PodToSelectableFields returns a field set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func PodToSelectableFields(pod *example.Pod) fields.Set {
	// The purpose of allocation with a given number of elements is to reduce
	// amount of allocations needed to create the fields.Set. If you add any
	// field here or the number of object-meta related fields changes, this should
	// be adjusted.
	podSpecificFieldsSet := make(fields.Set, 5)
	podSpecificFieldsSet["spec.nodeName"] = pod.Spec.NodeName
	podSpecificFieldsSet["spec.restartPolicy"] = string(pod.Spec.RestartPolicy)
	podSpecificFieldsSet["status.phase"] = string(pod.Status.Phase)
	return AddObjectMetaFieldsSet(podSpecificFieldsSet, &pod.ObjectMeta, true)
}

func AddObjectMetaFieldsSet(source fields.Set, objectMeta *metav1.ObjectMeta, hasNamespaceField bool) fields.Set {
	source["metadata.name"] = objectMeta.Name
	if hasNamespaceField {
		source["metadata.namespace"] = objectMeta.Namespace
	}
	return source
}

func testSetupWithEtcdServer(t *testing.T, opts ...setupOption) (context.Context, *cacher.Cacher, *etcd3testing.EtcdTestServer, tearDownFunc) {
	setupOpts := setupOptions{}
	opts = append([]setupOption{withDefaults}, opts...)
	for _, opt := range opts {
		opt(&setupOpts)
	}

	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	// Inject one list error to make sure we test the relist case.
	wrappedStorage := &storagetesting.StorageInjectingListErrors{
		Interface: etcdStorage,
		Errors:    1,
	}

	config := cacher.Config{
		Storage:        wrappedStorage,
		Versioner:      storage.APIObjectVersioner{},
		GroupResource:  schema.GroupResource{Resource: "pods"},
		ResourcePrefix: setupOpts.resourcePrefix,
		KeyFunc:        setupOpts.keyFunc,
		GetAttrsFunc:   GetPodAttrs,
		NewFunc:        newPod,
		NewListFunc:    newPodList,
		IndexerFuncs:   setupOpts.indexerFuncs,
		Codec:          codecs.LegacyCodec(examplev1.SchemeGroupVersion),
		Clock:          setupOpts.clock,
	}
	cacher, err := cacher.NewCacherFromConfig(config)
	if err != nil {
		t.Fatalf("Failed to initialize cacher: %v", err)
	}
	ctx := context.Background()
	terminate := func() {
		cacher.Stop()
		server.Terminate(t)
	}

	// Since some tests depend on the fact that GetList shouldn't fail,
	// we wait until the error from the underlying storage is consumed.
	if err := wait.PollInfinite(100*time.Millisecond, wrappedStorage.ErrorsConsumed); err != nil {
		t.Fatalf("Failed to inject list errors: %v", err)
	}

	return ctx, cacher, server, terminate
}

func testSetupWithEtcdAndCreateWrapper(t *testing.T, opts ...setupOption) (storage.Interface, tearDownFunc) {
	_, cacher, _, tearDown := testSetupWithEtcdServer(t, opts...)

	if err := cacher.ready.wait(context.TODO()); err != nil {
		t.Fatalf("unexpected error waiting for the cache to be ready")
	}
	return &createWrapper{Cacher: cacher}, tearDown
}

func newPod() runtime.Object     { return &example.Pod{} }
func newPodList() runtime.Object { return &example.PodList{} }

func newEtcdTestStorage(t *testing.T, prefix string) (*etcd3testing.EtcdTestServer, storage.Interface) {
	server, _ := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	storage := etcd3.New(
		server.V3Client,
		apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion),
		newPod,
		newPodList,
		prefix,
		"/pods",
		schema.GroupResource{Resource: "pods"},
		identity.NewEncryptCheckTransformer(),
		etcd3.NewDefaultLeaseManagerConfig())
	return server, storage
}

type createWrapper struct {
	*cacher.Cacher
}

func (c *createWrapper) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	if err := c.Cacher.Create(ctx, key, obj, out, ttl); err != nil {
		return err
	}
	return wait.PollUntilContextTimeout(ctx, 100*time.Millisecond, wait.ForeverTestTimeout, true, func(ctx context.Context) (bool, error) {
		currentObj := c.Cacher.newFunc()
		err := c.Cacher.Get(ctx, key, storage.GetOptions{ResourceVersion: "0"}, currentObj)
		if err != nil {
			if storage.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		if !apiequality.Semantic.DeepEqual(currentObj, out) {
			return false, nil
		}
		return true, nil
	})
}

type tearDownFunc func()

type setupOptions struct {
	resourcePrefix string
	keyFunc        func(runtime.Object) (string, error)
	indexerFuncs   map[string]storage.IndexerFunc
	clock          clock.WithTicker
}

type setupOption func(*setupOptions)

func withDefaults(options *setupOptions) {
	prefix := "/pods"

	options.resourcePrefix = prefix
	options.keyFunc = func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc(prefix, obj) }
	options.clock = clock.RealClock{}
}

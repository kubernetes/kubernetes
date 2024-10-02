/*
Copyright 2015 The Kubernetes Authors.

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

package storage_test

import (
	"context"
	"math/rand"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/api/apitesting"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	example2v1 "k8s.io/apiserver/pkg/apis/example2/v1"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
	utilruntime.Must(example2v1.AddToScheme(scheme))
}

func TestHighWaterMark(t *testing.T) {
	var h storage.HighWaterMark

	for i := int64(10); i < 20; i++ {
		if !h.Update(i) {
			t.Errorf("unexpected false for %v", i)
		}
		if h.Update(i - 1) {
			t.Errorf("unexpected true for %v", i-1)
		}
	}

	m := int64(0)
	wg := sync.WaitGroup{}
	for i := 0; i < 300; i++ {
		wg.Add(1)
		v := rand.Int63()
		go func(v int64) {
			defer wg.Done()
			h.Update(v)
		}(v)
		if v > m {
			m = v
		}
	}
	wg.Wait()
	if m != int64(h) {
		t.Errorf("unexpected value, wanted %v, got %v", m, int64(h))
	}
}

func TestGetCurrentResourceVersionFromStorage(t *testing.T) {
	// test data
	newEtcdTestStorage := func(t *testing.T, prefix string) (*etcd3testing.EtcdTestServer, storage.Interface) {
		server, _ := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
		storage := etcd3.New(server.V3Client, apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion, example2v1.SchemeGroupVersion), func() runtime.Object { return &example.Pod{} }, func() runtime.Object { return &example.PodList{} }, prefix, "/pods", schema.GroupResource{Resource: "pods"}, identity.NewEncryptCheckTransformer(), etcd3.NewDefaultLeaseManagerConfig())
		return server, storage
	}
	server, etcdStorage := newEtcdTestStorage(t, "")
	defer server.Terminate(t)
	versioner := storage.APIObjectVersioner{}

	makePod := func(name string) *example.Pod {
		return &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: name},
		}
	}
	createPod := func(obj *example.Pod) *example.Pod {
		key := "pods/" + obj.Namespace + "/" + obj.Name
		out := &example.Pod{}
		err := etcdStorage.Create(context.TODO(), key, obj, out, 0)
		require.NoError(t, err)
		return out
	}
	getPod := func(name, ns string) *example.Pod {
		key := "pods/" + ns + "/" + name
		out := &example.Pod{}
		err := etcdStorage.Get(context.TODO(), key, storage.GetOptions{}, out)
		require.NoError(t, err)
		return out
	}
	makeReplicaSet := func(name string) *example2v1.ReplicaSet {
		return &example2v1.ReplicaSet{
			ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: name},
		}
	}
	createReplicaSet := func(obj *example2v1.ReplicaSet) *example2v1.ReplicaSet {
		key := "replicasets/" + obj.Namespace + "/" + obj.Name
		out := &example2v1.ReplicaSet{}
		err := etcdStorage.Create(context.TODO(), key, obj, out, 0)
		require.NoError(t, err)
		return out
	}

	// create a pod and make sure its RV is equal to the one maintained by etcd
	pod := createPod(makePod("pod-1"))
	currentStorageRV, err := storage.GetCurrentResourceVersionFromStorage(context.TODO(), etcdStorage, func() runtime.Object { return &example.PodList{} }, "/pods", "Pod")
	require.NoError(t, err)
	podRV, err := versioner.ParseResourceVersion(pod.ResourceVersion)
	require.NoError(t, err)
	require.Equal(t, currentStorageRV, podRV, "expected the global etcd RV to be equal to pod's RV")

	// now create a replicaset (new resource) and make sure the target function returns global etcd RV
	rs := createReplicaSet(makeReplicaSet("replicaset-1"))
	currentStorageRV, err = storage.GetCurrentResourceVersionFromStorage(context.TODO(), etcdStorage, func() runtime.Object { return &example.PodList{} }, "/pods", "Pod")
	require.NoError(t, err)
	rsRV, err := versioner.ParseResourceVersion(rs.ResourceVersion)
	require.NoError(t, err)
	require.Equal(t, currentStorageRV, rsRV, "expected the global etcd RV to be equal to replicaset's RV")

	// ensure that the pod's RV hasn't been changed
	currentPod := getPod(pod.Name, pod.Namespace)
	currentPodRV, err := versioner.ParseResourceVersion(currentPod.ResourceVersion)
	require.NoError(t, err)
	require.Equal(t, currentPodRV, podRV, "didn't expect to see the pod's RV changed")
}

func TestHasInitialEventsEndBookmarkAnnotation(t *testing.T) {
	createPod := func(name string) *example.Pod {
		return &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: name}}
	}
	createAnnotatedPod := func(name, value string) *example.Pod {
		p := createPod(name)
		p.Annotations = map[string]string{}
		p.Annotations[metav1.InitialEventsAnnotationKey] = value
		return p
	}
	scenarios := []struct {
		name             string
		object           runtime.Object
		expectAnnotation bool
	}{
		{
			name:             "a standard obj with the initial-events-end annotation set to true",
			object:           createAnnotatedPod("p1", "true"),
			expectAnnotation: true,
		},
		{
			name:   "a standard obj with the initial-events-end annotation set to false",
			object: createAnnotatedPod("p1", "false"),
		},
		{
			name:   "a standard obj without the annotation",
			object: createPod("p1"),
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			hasAnnotation, err := storage.HasInitialEventsEndBookmarkAnnotation(scenario.object)
			require.NoError(t, err)
			require.Equal(t, scenario.expectAnnotation, hasAnnotation)
		})
	}
}

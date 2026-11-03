/*
Copyright The Kubernetes Authors.

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

	clientv3 "go.etcd.io/etcd/client/v3"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	example2v1 "k8s.io/apiserver/pkg/apis/example2/v1"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	etcdfeature "k8s.io/apiserver/pkg/storage/feature"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/value"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
	"k8s.io/utils/clock"
)

var (
	scheme = runtime.NewScheme()
	Codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
	utilruntime.Must(example2v1.AddToScheme(scheme))
	utilruntime.Must(corev1.AddToScheme(scheme))
	utilruntime.Must(metav1.AddMetaToScheme(scheme))
	scheme.AddUnversionedTypes(corev1.SchemeGroupVersion, &metav1.Status{})
	pb := protobuf.NewSerializer(scheme, scheme)
	Corev1ProtoCodec = Codecs.CodecForVersions(pb, pb, schema.GroupVersions{corev1.SchemeGroupVersion}, schema.GroupVersions{corev1.SchemeGroupVersion})
	Examplev1ProtoCodec = Codecs.CodecForVersions(pb, pb, schema.GroupVersions{examplev1.SchemeGroupVersion}, nil)
}

var (
	Corev1ProtoCodec    runtime.Codec
	Examplev1ProtoCodec runtime.Codec
)

func NewPod() runtime.Object     { return &example.Pod{} }
func NewPodList() runtime.Object { return &example.PodList{} }

func NewEtcdTestStorage(t testing.TB, prefix string) (*etcd3testing.EtcdTestServer, storage.Interface) {
	return NewEtcdTestStorageWithCodec(t, prefix, Examplev1ProtoCodec)
}

func NewEtcdTestStorageWithCodec(t testing.TB, prefix string, codec runtime.Codec) (*etcd3testing.EtcdTestServer, storage.Interface) {
	return NewEtcdTestStorageWithOptions(t, prefix, codec, identity.NewEncryptCheckTransformer(), storagetesting.PodReverseKeyFunc("/pods/", true))
}

func NewEtcdTestStorageWithOptions(t testing.TB, prefix string, codec runtime.Codec, transformer value.Transformer, reverseKeyFunc storage.ReverseKeyFunc) (*etcd3testing.EtcdTestServer, storage.Interface) {
	server, _ := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	versioner := storage.APIObjectVersioner{}
	compactor := etcd3.NewCompactor(server.V3Client.Client, 0, clock.RealClock{}, nil)
	t.Cleanup(compactor.Stop)
	storage, err := etcd3.New(
		server.V3Client,
		compactor,
		codec,
		NewPod,
		NewPodList,
		reverseKeyFunc,
		prefix,
		"/pods/",
		schema.GroupResource{Resource: "pods"},
		transformer,
		etcd3.NewDefaultLeaseManagerConfig(),
		etcd3.NewDefaultDecoder(codec, versioner),
		versioner)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(storage.Close)
	return server, storage
}

func ComputePodKey(obj metav1.Object) string {
	return fmt.Sprintf("/pods/%s/%s", obj.GetNamespace(), obj.GetName())
}

func IncreaseRVFunc(client *clientv3.Client) storagetesting.IncreaseRVFunc {
	return func(ctx context.Context, t *testing.T) int64 {
		resp, err := client.KV.Put(ctx, "increaseRV", "ok")
		if err != nil {
			t.Fatalf("Could not update increaseRV: %v", err)
		}
		return resp.Header.Revision
	}
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

// TODO(p0lyn0mial): ForceRequestWatchProgressSupport inits the storage layer
// so that tests that require storage.RequestWatchProgress pass
//
// In the future we could have a function that would allow for setting the feature
// only for duration of a test.
func ForceRequestWatchProgressSupport(t testing.TB) {
	if etcdfeature.DefaultFeatureSupportChecker.Supports(storage.RequestWatchProgress) {
		return
	}

	server, _ := NewEtcdTestStorage(t, etcd3testing.PathPrefix())
	defer server.Terminate(t)
	if err := wait.PollUntilContextTimeout(context.Background(), 5*time.Millisecond, wait.ForeverTestTimeout, true, func(_ context.Context) (bool, error) {
		return etcdfeature.DefaultFeatureSupportChecker.Supports(storage.RequestWatchProgress), nil
	}); err != nil {
		t.Fatalf("failed to wait for required %v storage feature to initialize", storage.RequestWatchProgress)
	}
}

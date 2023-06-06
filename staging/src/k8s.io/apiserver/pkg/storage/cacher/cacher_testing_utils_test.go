/*
Copyright 2023 The Kubernetes Authors.

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

package cacher

import (
	"fmt"
	"testing"

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
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
)

var (
	scheme   = runtime.NewScheme()
	codecs   = serializer.NewCodecFactory(scheme)
	errDummy = fmt.Errorf("dummy error")
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
	utilruntime.Must(example2v1.AddToScheme(scheme))
}

func newPod() runtime.Object     { return &example.Pod{} }
func newPodList() runtime.Object { return &example.PodList{} }

func newEtcdTestStorage(t *testing.T, prefix string, pagingEnabled bool) (*etcd3testing.EtcdTestServer, storage.Interface) {
	server, _ := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	storage := etcd3.New(
		server.V3Client,
		apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion),
		newPod,
		prefix,
		schema.GroupResource{Resource: "pods"},
		identity.NewEncryptCheckTransformer(),
		pagingEnabled,
		etcd3.NewDefaultLeaseManagerConfig())
	return server, storage
}

func makeTestPodWithName(name string) *example.Pod {
	return &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: name},
		Spec:       storagetesting.DeepEqualSafePodSpec(),
	}
}

func computePodKey(obj *example.Pod) string {
	return fmt.Sprintf("/pods/%s/%s", obj.Namespace, obj.Name)
}

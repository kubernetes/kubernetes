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

package apiserver

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"testing"

	"golang.org/x/net/http2"

	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	appsv1applyconfigurations "k8s.io/client-go/applyconfigurations/apps/v1"
	clientset "k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestRequestObjectConvertibleToUnstructured tests that write requests fail if the request object
// is not convertible to unstructured. The ability to convert an object to unstructured ensures that
// it can be encoded as JSON and that field managers can be determined.
func TestRequestObjectConvertibleToUnstructured(t *testing.T) {
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{}, framework.SharedEtcd())
	defer server.TearDownFn()

	for i, raw := range []string{
		``,
		`"`,
		`{`,
		`[`,
		`1z`,
		`z`,
	} {
		// The Protobuf request encoding is required. Invalid JSON cannot be embedded in a
		// JSON object or array without making the containing object or array also invalid.
		protoConfig := server.ClientConfig
		protoConfig.ContentConfig.ContentType = runtime.ContentTypeProtobuf
		protoConfig.ContentConfig.AcceptContentTypes = runtime.ContentTypeProtobuf
		protoClient, err := clientset.NewForConfig(protoConfig)
		if err != nil {
			t.Fatalf("unexpected error creating proto client: %v", err)
		}

		createError := new(http2.StreamError)
		if _, err := protoClient.AppsV1().ControllerRevisions("default").Create(context.TODO(), &appsv1.ControllerRevision{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("test-revision-create-%d", i),
			},
			Data: runtime.RawExtension{Raw: []byte(raw)},
		}, metav1.CreateOptions{}); errors.As(err, createError) && createError.Code == http2.ErrCodeInternal {
			t.Logf("create returned internal error as expected with rawextension %#v: %v", raw, err)
		} else {
			t.Errorf("create returned unexpected error: %#v", err)
		}

		var marshalerError *json.MarshalerError
		if _, err := protoClient.AppsV1().ControllerRevisions("default").Apply(context.TODO(), appsv1applyconfigurations.ControllerRevision("test-revision-apply", "default").
			WithData(runtime.RawExtension{Raw: []byte(raw)}),
			metav1.ApplyOptions{}); errors.As(err, &marshalerError) {
			// In this case the error is currently client-side, since apply request
			// bodies must be encoded as JSON. Included here to cover the future
			// possibility of Protobuf-encoded apply configurations.
			t.Logf("apply returned client-side marshaler error as expected with rawextension %#v: %v", raw, err)
		} else {
			t.Errorf("apply returned unexpected error: %#v", err)
		}

		// Create an object to be updated. If the object does not exist, then the update
		// will short-circuit on "not found" before it encounters the error that is
		// interesting to this test.
		existing, err := protoClient.AppsV1().ControllerRevisions("default").Create(context.TODO(), &appsv1.ControllerRevision{
			ObjectMeta: metav1.ObjectMeta{
				Name: fmt.Sprintf("test-revision-update-%d", i),
			},
			Data: runtime.RawExtension{Raw: []byte(`{}`)},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Errorf("expected nil create error, got: %v", err)
			continue
		}

		updateError := new(http2.StreamError)
		existing.Data = runtime.RawExtension{Raw: []byte(raw)}
		if _, err := protoClient.AppsV1().ControllerRevisions(existing.Namespace).Update(context.TODO(), existing, metav1.UpdateOptions{}); errors.As(err, updateError) && updateError.Code == http2.ErrCodeInternal {
			t.Logf("update returned internal error as expected with rawextension %#v: %v", raw, err)
		} else {
			t.Errorf("update returned unexpected error: %#v", err)
		}
	}
}

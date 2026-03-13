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
	"bytes"
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/direct"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
	"k8s.io/client-go/kubernetes/scheme"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

// TestNondeterministicResponseEncoding verifies that the encoding of response bodies to CBOR is not
// deterministic. Even in cases where encoding deterministically has no overhead, some randomness is
// introduced to prevent clients from inadvertently depending on deterministic encoding when it is
// not guaranteed.
func TestNondeterministicResponseEncoding(t *testing.T) {
	// Nondeterministic map key order is not guaranteed to select each possible ordering with
	// equal probability. In practice, since metav1.WatchEvent only has two fields, it does and
	// the probability of either possible map key ordering is approximately 50%. The probability
	// that this test flakes because a watch event was encoded the same way on every trial is
	// about 2^-NTrials, so NTrials needs to be big enough to make sure that doesn't happen.
	const NTrials = 40

	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.CBORServingAndStorage, true)
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientsAllowCBOR, true)
	clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.ClientsPreferCBOR, true)

	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	t.Cleanup(server.TearDownFn)

	config := rest.CopyConfig(server.ClientConfig)
	config.AcceptContentTypes = "application/cbor"
	client, err := corev1client.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	namespace, err := client.Namespaces().Create(context.TODO(), &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "test-nondeterministic-response-encoding",
			Annotations: map[string]string{"hello": "world"},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// Compare pairs of "get" requests at the same resource version for encoding differences.
	responseDiff := false
	for i := 0; i < NTrials && !responseDiff; i++ {
		request := client.RESTClient().Get().Resource("namespaces").Name(namespace.GetName())

		// get at latest resource version
		result := request.Do(context.TODO())
		raw1, err := result.Raw()
		if err != nil {
			t.Fatal(err)
		}
		if err := result.Into(namespace); err != nil {
			t.Fatal(err)
		}

		// get again at same resource version
		trial := request.VersionedParams(&metav1.GetOptions{ResourceVersion: namespace.ResourceVersion}, scheme.ParameterCodec).Do(context.TODO())
		var trialObject corev1.Namespace
		if err := trial.Into(&trialObject); err != nil {
			if errors.IsResourceExpired(err) {
				t.Logf("retrying: %v", err)
				continue
			}
			t.Fatal(err)
		}
		if !equality.Semantic.DeepEqual(namespace, &trialObject) {
			t.Fatalf("objects differed semantically between runs:\n%s", cmp.Diff(namespace, trialObject))
		}
		raw2, err := trial.Raw()
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(raw1, raw2) {
			// Observed a response body that was not byte-for-byte the same as the first
			// response body.
			responseDiff = true
		}
	}
	if !responseDiff {
		t.Errorf("performed %d consecutive get requests to the same resource and observed identical response bodies each time", NTrials)
	}

	// Induce a watch event, then compare the encoding of the induced event across pairs of
	// watch requests.
	eventDiff := false
	objDiff := false
	for i := 0; i < NTrials && (!eventDiff || !objDiff); i++ {
		// Get latest to have a valid resource version to start watches from.
		namespace, err := client.Namespaces().Get(context.TODO(), namespace.GetName(), metav1.GetOptions{})
		if err != nil {
			t.Fatal(err)
		}

		// Patch so that watchers will see a "modified" event.
		patched, err := client.Namespaces().Patch(context.TODO(), namespace.GetName(), types.JSONPatchType, []byte(fmt.Sprintf(`[{"op":"add","path":"/metadata/annotations/foo","value":"%d"}]`, i)), metav1.PatchOptions{})
		if err != nil {
			t.Fatal(err)
		}

		request := client.RESTClient().Get().Resource("namespaces")

		// Get the raw bytes of the watch event induced by the patch plus the raw bytes of
		// its embedded object.
		getRawEventAndRawObject := func() ([]byte, []byte, error) {
			ctx, cancel := context.WithTimeout(context.TODO(), 6*time.Second)
			defer cancel()
			rc, err := request.
				VersionedParams(&metav1.ListOptions{
					ResourceVersion: namespace.ResourceVersion,
					Watch:           true,
					TimeoutSeconds:  ptr.To(int64(5)),
					FieldSelector:   fmt.Sprintf("metadata.name=%s", namespace.GetName()),
				}, scheme.ParameterCodec).
				Stream(ctx)
			if err != nil {
				return nil, nil, err
			}
			defer func() {
				if err := rc.Close(); err != nil {
					t.Error(err)
				}
			}()
			d := &rawCapturingDecoder{delegate: cbor.NewSerializer(scheme.Scheme, scheme.Scheme, cbor.Transcode(false))}
			sd := streaming.NewDecoder(rc, d)
			for {
				var event metav1.WatchEvent
				got, _, err := sd.Decode(nil, &event)
				if err != nil {
					// Either the server timeout or client context timeout will
					// cause an EOF here to terminate the loop if the expected
					// event is never received.
					t.Fatal(err)
				}
				if got != &event {
					t.Fatalf("returned new object %#v (%T) instead of decoding into %T", got, got, &event)
				}
				var u map[string]interface{}
				if err := direct.Unmarshal(event.Object.Raw, &u); err != nil {
					t.Fatal(err)
				}
				rv, ok, err := unstructured.NestedString(u, "metadata", "resourceVersion")
				if err != nil {
					t.Fatalf("failed to get resourceVersion from watch event: %v", err)
				}
				if !ok {
					t.Fatal("watch event missing resource version")
				}
				if rv == patched.ResourceVersion {
					return d.raw, event.Object.Raw, nil
				}
			}
		}

		event1, raw1, err := getRawEventAndRawObject()
		if err != nil {
			if errors.IsResourceExpired(err) {
				t.Logf("retrying: %v", err)
				continue
			}
			t.Fatal(err)
		}
		var obj1 map[string]interface{}
		if err := direct.Unmarshal(raw1, &obj1); err != nil {
			t.Fatal(err)
		}

		event2, raw2, err := getRawEventAndRawObject()
		if err != nil {
			if errors.IsResourceExpired(err) {
				t.Logf("retrying: %v", err)
				continue
			}
			t.Fatal(err)
		}
		var obj2 map[string]interface{}
		if err := direct.Unmarshal(raw2, &obj2); err != nil {
			t.Fatal(err)
		}

		if !equality.Semantic.DeepEqual(obj1, obj2) {
			t.Fatalf("objects differed semantically between runs:\n%s", cmp.Diff(obj1, obj2))
		}

		if !bytes.Equal(raw1, raw2) {
			objDiff = true
		}

		// Cut out the embedded object so that we can observe that the watch event itself is
		// encoded nondeterministically rather than simply embedding a nondeterministic
		// object encoding.
		event1 = bytes.Replace(event1, raw1, nil, 1)
		event2 = bytes.Replace(event2, raw2, nil, 1)
		if !bytes.Equal(event1, event2) {
			eventDiff = true
		}
	}
	if !eventDiff {
		t.Errorf("watch event encoded identically over %d consecutive watch requests", NTrials)
	}
	if !objDiff {
		t.Errorf("watch event embedded object encoded identically over %d consecutive watch requests", NTrials)
	}
}

type rawCapturingDecoder struct {
	raw      []byte
	delegate runtime.Decoder
}

func (d *rawCapturingDecoder) Decode(data []byte, defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	d.raw = append([]byte(nil), data...)
	return d.delegate.Decode(data, defaults, into)
}

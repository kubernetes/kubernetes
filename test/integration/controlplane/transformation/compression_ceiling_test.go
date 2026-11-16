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

package transformation

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"errors"
	"testing"
	"time"

	apiextensionsclientset "k8s.io/apiextensions-api/pkg/client/clientset/clientset"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/dynamic"
	"k8s.io/kubernetes/test/integration/etcd"
	"k8s.io/kubernetes/test/integration/framework"
)

// etcdDefaultMaxRequestBytes is etcd's default --max-request-bytes. Neither
// hack/lib/etcd.sh nor framework.RunCustomEtcd sets that flag, so it's the
// limit the integration etcd enforces.
const etcdDefaultMaxRequestBytes = 1536 * 1024

const etcdRequestErrorsMetric = "etcd_request_errors_total"

// newFoo returns a Foo whose data field holds n bytes of compressible text.
// ConfigMaps and Secrets cap their data at 1MiB, so reaching the ceiling would
// mean padding their metadata, while a Foo gets there with one field.
func newFoo(name string, n int) *unstructured.Unstructured {
	return &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "cr.bar.com/v1",
		"kind":       "Foo",
		"metadata":   map[string]any{"name": name},
		"data":       compressibleText(n),
	}}
}

// declaredLength returns the plaintext length that a deflate frame declares
// in the uvarint after its header.
func declaredLength(t *testing.T, frame []byte) uint64 {
	t.Helper()
	n, size := binary.Uvarint(frame[frameHeaderSize:])
	if size <= 0 {
		t.Fatalf("the frame's declared length doesn't parse: %x", frame[frameHeaderSize:min(len(frame), frameHeaderSize+4)])
	}
	return n
}

// apiFailure is the code and reason of an API error, so that two failures can
// be compared without their messages.
type apiFailure struct {
	code   int32
	reason metav1.StatusReason
}

func failureOf(t *testing.T, err error) apiFailure {
	t.Helper()
	var status apierrors.APIStatus
	if !errors.As(err, &status) {
		t.Fatalf("got %v, want an API status error", err)
	}
	return apiFailure{code: status.Status().Code, reason: status.Status().Reason}
}

func wantFooData(t *testing.T, foos dynamic.ResourceInterface, want *unstructured.Unstructured) {
	t.Helper()
	got, err := foos.Get(testContext(t), want.GetName(), metav1.GetOptions{})
	if err != nil {
		t.Fatalf("failed to read foo %s: %v", want.GetName(), err)
	}
	if got.Object["data"] != want.Object["data"] {
		t.Errorf("foo %s read back with different data", want.GetName())
	}
}

// TestStorageCompressionCeiling stores Foos around the ceiling with foos
// compressed, then restarts without the policy. A value at the ceiling is
// compressed and still fits etcd once it's rewritten bare. Anything bigger is
// stored, or refused, just as it would be with compression off, even when it
// would compress well.
func TestStorageCompressionCeiling(t *testing.T) {
	storageConfig := framework.SharedEtcd()
	policy := compressionFlags(t, compressionConfig(compressionEntry(fooGVR.GroupResource().String(), "Deflate", "")), "1.38")
	start := func(t *testing.T, flags []string) (*transformTest, dynamic.ResourceInterface) {
		t.Helper()
		test, err := newTransformTest(t, transformTestConfig{storageConfig: storageConfig, extraFlags: flags})
		if err != nil {
			t.Fatalf("failed to start kube-apiserver: %v", err)
		}
		return test, dynamic.NewForConfigOrDie(test.kubeAPIServer.ClientConfig).Resource(fooGVR).Namespace(testNamespace)
	}
	create := func(t *testing.T, foos dynamic.ResourceInterface, foo *unstructured.Unstructured) error {
		t.Helper()
		_, err := foos.Create(testContext(t), foo, metav1.CreateOptions{})
		return err
	}
	// The data alone fills etcd's request limit, so etcd refuses it whatever
	// else is in the request.
	tooLarge := newFoo("too-large", etcdDefaultMaxRequestBytes)
	// createTooLarge returns how creating too-large failed. The apiserver turns
	// etcd's refusal into a generic 500, so only the etcd error counter shows
	// that etcd refused it. InvalidArgument is etcd's code for a request over
	// its limit.
	createTooLarge := func(t *testing.T, test *transformTest, foos dynamic.ResourceInterface) apiFailure {
		t.Helper()
		refusals := map[string]string{"operation": "create", "group": fooGVR.Group, "resource": fooGVR.Resource, "reason": "InvalidArgument"}
		before := counterValue(t, etcdRequestErrorsMetric, refusals)
		err := create(t, foos, tooLarge)
		if err == nil {
			t.Fatalf("foo %s was stored, so etcd isn't at its default %d byte request limit", tooLarge.GetName(), etcdDefaultMaxRequestBytes)
		}
		if counterValue(t, etcdRequestErrorsMetric, refusals) == before {
			t.Errorf("creating foo %s failed, but %s doesn't show etcd refusing it with InvalidArgument", tooLarge.GetName(), etcdRequestErrorsMetric)
		}
		if got := test.rawValue(t, fooGVR.GroupResource(), testNamespace, tooLarge.GetName()); got != nil {
			t.Errorf("foo %s was refused but etcd holds %d bytes for it", tooLarge.GetName(), len(got))
		}
		return failureOf(t, err)
	}
	// The first run sets these, and the second checks the same Foos after the restart.
	var atCeiling, overCeiling *unstructured.Unstructured
	var tooLargeFailure apiFailure

	// The second run checks what the first one stored, so stop if the first fails.
	if !t.Run("foos compressed", func(t *testing.T) {
		test, foos := start(t, policy)
		defer test.cleanUp()
		// Foo is a custom resource, so its CRD has to exist before any Foo does.
		crd := etcd.GetCustomResourceDefinitionData()[0]
		etcd.CreateTestCRDs(t, apiextensionsclientset.NewForConfigOrDie(test.kubeAPIServer.ClientConfig), false, crd)

		// Measure the bytes every Foo shares by storing one with no data. It's
		// below the 1Ki floor, so etcd holds it bare. Foos differ only in name and
		// data, so this sizes each later Foo to the exact byte.
		if err := create(t, foos, newFoo("probe", 0)); err != nil {
			t.Fatalf("failed to create foo probe: %v", err)
		}
		probe := test.rawValue(t, fooGVR.GroupResource(), testNamespace, "probe")
		if got := formOf(probe); got != formUnframedJSON {
			t.Fatalf("foo probe is stored as %s, want %s", got, formUnframedJSON)
		}
		// A Foo is stored as JSON, where its name and data each appear once and
		// need no escaping. So it takes overhead + len(name) + len(data) bytes, and
		// ofSize picks the data length that makes the total exactly size.
		overhead := len(probe) - len("probe")
		ofSize := func(name string, size int) *unstructured.Unstructured {
			return newFoo(name, size-overhead-len(name))
		}
		// at-ceiling is the largest value the policy frames, and over-ceiling the
		// smallest it doesn't.
		atCeiling = ofSize("at-ceiling", maxPlaintextBytes)
		overCeiling = ofSize("over-ceiling", maxPlaintextBytes+1)

		// at-ceiling is compressed, and its frame declares the full ceiling.
		if err := create(t, foos, atCeiling); err != nil {
			t.Fatalf("failed to create foo %s: %v", atCeiling.GetName(), err)
		}
		raw := test.rawValue(t, fooGVR.GroupResource(), testNamespace, atCeiling.GetName())
		if got := formOf(raw); got != formDeflate {
			t.Fatalf("foo %s is stored as %s, want %s", atCeiling.GetName(), got, formDeflate)
		}
		if got := declaredLength(t, raw); got != maxPlaintextBytes {
			t.Fatalf("foo %s's frame declares %d bytes, want the ceiling of %d", atCeiling.GetName(), got, maxPlaintextBytes)
		}

		// The outcome counter says it was the ceiling, not the floor or the
		// policy, that kept the value bare.
		aboveMax := map[string]string{"resource": fooGVR.GroupResource().String(), "operation": "to_storage", "outcome": "unframed_above_max"}
		aboveMaxBefore := counterValue(t, operationsMetric, aboveMax)
		if err := create(t, foos, overCeiling); err != nil {
			t.Fatalf("failed to create foo %s: %v", overCeiling.GetName(), err)
		}
		if counterValue(t, operationsMetric, aboveMax) == aboveMaxBefore {
			t.Errorf("%s didn't count foo %s as above the ceiling", operationsMetric, overCeiling.GetName())
		}
		// over-ceiling is stored bare, byte for byte, and a no-op update leaves it unchanged.
		raw = test.rawValue(t, fooGVR.GroupResource(), testNamespace, overCeiling.GetName())
		if got := formOf(raw); got != formUnframedJSON || len(raw) != maxPlaintextBytes+1 {
			t.Fatalf("foo %s is stored as %d bytes of %s, want %d bytes of %s", overCeiling.GetName(), len(raw), got, maxPlaintextBytes+1, formUnframedJSON)
		}
		if test.noOpUpdate(t, fooGVR, overCeiling.GetName()) {
			t.Errorf("a no-op update of foo %s changed its resource version, but the policy can't frame it", overCeiling.GetName())
		}

		// DEFLATE would shrink too-large to a few KiB, but it's above the ceiling,
		// so etcd gets it bare and refuses it. The body has to fit the apiserver's
		// own limit, 3MiB by default, or the apiserver would refuse it before etcd
		// saw it.
		body, err := json.Marshal(tooLarge.Object)
		if err != nil {
			t.Fatal(err)
		}
		if limit := test.kubeAPIServer.ServerOpts.GenericServerRunOptions.MaxRequestBodyBytes; int64(len(body)) >= limit {
			t.Fatalf("foo %s's body is %d bytes, want it under the apiserver's %d byte limit", tooLarge.GetName(), len(body), limit)
		}
		tooLargeFailure = createTooLarge(t, test, foos)
		// Both Foos that were stored read back with the data they were created with.
		wantFooData(t, foos, atCeiling)
		wantFooData(t, foos, overCeiling)
	}) {
		return
	}

	t.Run("compression removed", func(t *testing.T) {
		test, foos := start(t, nil)
		defer test.cleanUp()
		// The CRD is already in etcd. Wait for this apiserver to serve it.
		crd := etcd.GetCustomResourceDefinitionData()[0]
		crdClient := apiextensionsclientset.NewForConfigOrDie(test.kubeAPIServer.ClientConfig)
		if err := wait.PollUntilContextTimeout(testContext(t), 100*time.Millisecond, wait.ForeverTestTimeout, true, func(context.Context) (bool, error) {
			return etcd.CrdExistsInDiscovery(crdClient, crd), nil
		}); err != nil {
			t.Fatalf("foos never showed up in discovery: %v", err)
		}

		test.wantForm(t, fooGVR, atCeiling.GetName(), formDeflate)
		wantFooData(t, foos, atCeiling)
		// The rewrite puts the whole value in one etcd request, and noOpUpdate
		// fails the test if etcd refuses it.
		test.wantRewriteOnce(t, fooGVR, atCeiling.GetName(), formUnframedJSON)
		if raw := test.rawValue(t, fooGVR.GroupResource(), testNamespace, atCeiling.GetName()); len(raw) != maxPlaintextBytes {
			t.Errorf("foo %s is stored as %d bytes, want the ceiling of %d", atCeiling.GetName(), len(raw), maxPlaintextBytes)
		}
		wantFooData(t, foos, atCeiling)

		// over-ceiling is stored bare, as compression off would store it, so
		// there's nothing to rewrite.
		if test.noOpUpdate(t, fooGVR, overCeiling.GetName()) {
			t.Errorf("a no-op update of foo %s changed its resource version", overCeiling.GetName())
		}
		wantFooData(t, foos, overCeiling)
		// With compression off, too-large has to fail the same way it did with
		// compression on.
		if got := createTooLarge(t, test, foos); got != tooLargeFailure {
			t.Errorf("creating foo %s failed with %+v, want %+v as with compression on", tooLarge.GetName(), got, tooLargeFailure)
		}
	})
}

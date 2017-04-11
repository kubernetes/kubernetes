/*
Copyright 2014 The Kubernetes Authors.

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

package api_test

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"io/ioutil"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/golang/protobuf/proto"
	"github.com/ugorji/go/codec"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	apitesting "k8s.io/apimachinery/pkg/api/testing"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	kapitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

// fuzzInternalObject fuzzes an arbitrary runtime object using the appropriate
// fuzzer registered with the apitesting package.
func fuzzInternalObject(t *testing.T, forVersion schema.GroupVersion, item runtime.Object, seed int64) runtime.Object {
	apitesting.FuzzerFor(kapitesting.FuzzerFuncs(t, api.Codecs), rand.NewSource(seed)).Fuzz(item)

	j, err := meta.TypeAccessor(item)
	if err != nil {
		t.Fatalf("Unexpected error %v for %#v", err, item)
	}
	j.SetKind("")
	j.SetAPIVersion("")

	return item
}

// dataAsString returns the given byte array as a string; handles detecting
// protocol buffers.
func dataAsString(data []byte) string {
	dataString := string(data)
	if !strings.HasPrefix(dataString, "{") {
		dataString = "\n" + hex.Dump(data)
		proto.NewBuffer(make([]byte, 0, 1024)).DebugPrint("decoded object", data)
	}
	return dataString
}

func Convert_v1beta1_ReplicaSet_to_api_ReplicationController(in *v1beta1.ReplicaSet, out *api.ReplicationController, s conversion.Scope) error {
	intermediate1 := &extensions.ReplicaSet{}
	if err := v1beta1.Convert_v1beta1_ReplicaSet_To_extensions_ReplicaSet(in, intermediate1, s); err != nil {
		return err
	}

	intermediate2 := &v1.ReplicationController{}
	if err := v1.Convert_extensions_ReplicaSet_to_v1_ReplicationController(intermediate1, intermediate2, s); err != nil {
		return err
	}

	return v1.Convert_v1_ReplicationController_To_api_ReplicationController(intermediate2, out, s)
}

func TestSetControllerConversion(t *testing.T) {
	if err := api.Scheme.AddConversionFuncs(Convert_v1beta1_ReplicaSet_to_api_ReplicationController); err != nil {
		t.Fatal(err)
	}

	rs := &extensions.ReplicaSet{}
	rc := &api.ReplicationController{}

	extGroup := testapi.Extensions
	defaultGroup := testapi.Default

	fuzzInternalObject(t, extGroup.InternalGroupVersion(), rs, rand.Int63())

	t.Logf("rs._internal.extensions -> rs.v1beta1.extensions")
	data, err := runtime.Encode(extGroup.Codec(), rs)
	if err != nil {
		t.Fatalf("unexpected encoding error: %v", err)
	}

	decoder := api.Codecs.DecoderToVersion(
		api.Codecs.UniversalDeserializer(),
		runtime.NewMultiGroupVersioner(
			*defaultGroup.GroupVersion(),
			schema.GroupKind{Group: defaultGroup.GroupVersion().Group},
			schema.GroupKind{Group: extGroup.GroupVersion().Group},
		),
	)

	t.Logf("rs.v1beta1.extensions -> rc._internal")
	if err := runtime.DecodeInto(decoder, data, rc); err != nil {
		t.Fatalf("unexpected decoding error: %v", err)
	}

	t.Logf("rc._internal -> rc.v1")
	data, err = runtime.Encode(defaultGroup.Codec(), rc)
	if err != nil {
		t.Fatalf("unexpected encoding error: %v", err)
	}

	t.Logf("rc.v1 -> rs._internal.extensions")
	if err := runtime.DecodeInto(decoder, data, rs); err != nil {
		t.Fatalf("unexpected decoding error: %v", err)
	}
}

// TestSpecificKind round-trips a single specific kind and is intended to help
// debug issues that arise while adding a new API type.
func TestSpecificKind(t *testing.T) {
	// Uncomment the following line to enable logging of which conversions
	// api.scheme.Log(t)
	internalGVK := schema.GroupVersionKind{Group: "extensions", Version: runtime.APIVersionInternal, Kind: "DaemonSet"}

	seed := rand.Int63()
	fuzzer := apitesting.FuzzerFor(kapitesting.FuzzerFuncs(t, api.Codecs), rand.NewSource(seed))

	apitesting.RoundTripSpecificKind(t, internalGVK, api.Scheme, api.Codecs, fuzzer, nil)
}

var nonRoundTrippableTypes = sets.NewString(
	"ExportOptions",
	"GetOptions",
	// WatchEvent does not include kind and version and can only be deserialized
	// implicitly (if the caller expects the specific object). The watch call defines
	// the schema by content type, rather than via kind/version included in each
	// object.
	"WatchEvent",
	// ListOptions is now part of the meta group
	"ListOptions",
	// Delete options is only read in metav1
	"DeleteOptions",
)

var commonKinds = []string{"Status", "ListOptions", "DeleteOptions", "ExportOptions"}

// TestCommonKindsRegistered verifies that all group/versions registered with
// the testapi package have the common kinds.
func TestCommonKindsRegistered(t *testing.T) {
	for _, kind := range commonKinds {
		for _, group := range testapi.Groups {
			gv := group.GroupVersion()
			gvk := gv.WithKind(kind)
			obj, err := api.Scheme.New(gvk)
			if err != nil {
				t.Error(err)
			}
			defaults := gv.WithKind("")
			var got *schema.GroupVersionKind
			if obj, got, err = api.Codecs.LegacyCodec().Decode([]byte(`{"kind":"`+kind+`"}`), &defaults, obj); err != nil || gvk != *got {
				t.Errorf("expected %v: %v %v", gvk, got, err)
			}
			data, err := runtime.Encode(api.Codecs.LegacyCodec(*gv), obj)
			if err != nil {
				t.Errorf("expected %v: %v\n%s", gvk, err, string(data))
				continue
			}
			if !bytes.Contains(data, []byte(`"kind":"`+kind+`","apiVersion":"`+gv.String()+`"`)) {
				if kind != "Status" {
					t.Errorf("expected %v: %v\n%s", gvk, err, string(data))
					continue
				}
				// TODO: this is wrong, but legacy clients expect it
				if !bytes.Contains(data, []byte(`"kind":"`+kind+`","apiVersion":"v1"`)) {
					t.Errorf("expected %v: %v\n%s", gvk, err, string(data))
					continue
				}
			}
		}
	}
}

// TestRoundTripTypes applies the round-trip test to all round-trippable Kinds
// in all of the API groups registered for test in the testapi package.
func TestRoundTripTypes(t *testing.T) {
	seed := rand.Int63()
	fuzzer := apitesting.FuzzerFor(kapitesting.FuzzerFuncs(t, api.Codecs), rand.NewSource(seed))

	nonRoundTrippableTypes := map[schema.GroupVersionKind]bool{
		{Group: "componentconfig", Version: runtime.APIVersionInternal, Kind: "KubeletConfiguration"}:       true,
		{Group: "componentconfig", Version: runtime.APIVersionInternal, Kind: "KubeProxyConfiguration"}:     true,
		{Group: "componentconfig", Version: runtime.APIVersionInternal, Kind: "KubeSchedulerConfiguration"}: true,
	}

	apitesting.RoundTripTypes(t, api.Scheme, api.Codecs, fuzzer, nonRoundTrippableTypes)
}

// TestEncodePtr tests that a pointer to a golang type can be encoded and
// decoded without information loss or mutation.
func TestEncodePtr(t *testing.T) {
	grace := int64(30)
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{"name": "foo"},
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,

			TerminationGracePeriodSeconds: &grace,

			SecurityContext: &api.PodSecurityContext{},
			SchedulerName:   api.DefaultSchedulerName,
		},
	}
	obj := runtime.Object(pod)
	data, err := runtime.Encode(testapi.Default.Codec(), obj)
	obj2, err2 := runtime.Decode(testapi.Default.Codec(), data)
	if err != nil || err2 != nil {
		t.Fatalf("Failure: '%v' '%v'", err, err2)
	}
	if _, ok := obj2.(*api.Pod); !ok {
		t.Fatalf("Got wrong type")
	}
	if !apiequality.Semantic.DeepEqual(obj2, pod) {
		t.Errorf("\nExpected:\n\n %#v,\n\nGot:\n\n %#vDiff: %v\n\n", pod, obj2, diff.ObjectDiff(obj2, pod))
	}
}

// TestBadJSONRejection establishes that a JSON object without a kind or with
// an unknown kind will not be decoded without error.
func TestBadJSONRejection(t *testing.T) {
	badJSONMissingKind := []byte(`{ }`)
	if _, err := runtime.Decode(testapi.Default.Codec(), badJSONMissingKind); err == nil {
		t.Errorf("Did not reject despite lack of kind field: %s", badJSONMissingKind)
	}
	badJSONUnknownType := []byte(`{"kind": "bar"}`)
	if _, err1 := runtime.Decode(testapi.Default.Codec(), badJSONUnknownType); err1 == nil {
		t.Errorf("Did not reject despite use of unknown type: %s", badJSONUnknownType)
	}
	/*badJSONKindMismatch := []byte(`{"kind": "Pod"}`)
	if err2 := DecodeInto(badJSONKindMismatch, &Node{}); err2 == nil {
		t.Errorf("Kind is set but doesn't match the object type: %s", badJSONKindMismatch)
	}*/
}

// TestUnversionedTypes establishes that the default codec can encode and
// decode unversioned objects.
func TestUnversionedTypes(t *testing.T) {
	testcases := []runtime.Object{
		&metav1.Status{Status: "Failure", Message: "something went wrong"},
		&metav1.APIVersions{Versions: []string{"A", "B", "C"}},
		&metav1.APIGroupList{Groups: []metav1.APIGroup{{Name: "mygroup"}}},
		&metav1.APIGroup{Name: "mygroup"},
		&metav1.APIResourceList{GroupVersion: "mygroup/myversion"},
	}

	for _, obj := range testcases {
		// Make sure the unversioned codec can encode
		unversionedJSON, err := runtime.Encode(testapi.Default.Codec(), obj)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", obj, err)
			continue
		}

		// Make sure the versioned codec under test can decode
		versionDecodedObject, err := runtime.Decode(testapi.Default.Codec(), unversionedJSON)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", obj, err)
			continue
		}
		// Make sure it decodes correctly
		if !reflect.DeepEqual(obj, versionDecodedObject) {
			t.Errorf("%v: expected %#v, got %#v", obj, obj, versionDecodedObject)
			continue
		}
	}
}

// TestObjectWatchFraming establishes that a watch event can be encoded and
// decoded correctly through each of the supported RFC2046 media types.
func TestObjectWatchFraming(t *testing.T) {
	f := apitesting.FuzzerFor(kapitesting.FuzzerFuncs(t, api.Codecs), rand.NewSource(benchmarkSeed))
	secret := &api.Secret{}
	f.Fuzz(secret)
	secret.Data["binary"] = []byte{0x00, 0x10, 0x30, 0x55, 0xff, 0x00}
	secret.Data["utf8"] = []byte("a string with \u0345 characters")
	secret.Data["long"] = bytes.Repeat([]byte{0x01, 0x02, 0x03, 0x00}, 1000)
	converted, _ := api.Scheme.ConvertToVersion(secret, v1.SchemeGroupVersion)
	v1secret := converted.(*v1.Secret)
	for _, info := range api.Codecs.SupportedMediaTypes() {
		if info.StreamSerializer == nil {
			continue
		}
		s := info.StreamSerializer
		framer := s.Framer
		embedded := info.Serializer
		if embedded == nil {
			t.Errorf("no embedded serializer for %s", info.MediaType)
			continue
		}
		innerDecode := api.Codecs.DecoderToVersion(embedded, api.SchemeGroupVersion)

		// write a single object through the framer and back out
		obj := &bytes.Buffer{}
		if err := s.Encode(v1secret, obj); err != nil {
			t.Fatal(err)
		}
		out := &bytes.Buffer{}
		w := framer.NewFrameWriter(out)
		if n, err := w.Write(obj.Bytes()); err != nil || n != len(obj.Bytes()) {
			t.Fatal(err)
		}
		sr := streaming.NewDecoder(framer.NewFrameReader(ioutil.NopCloser(out)), s)
		resultSecret := &v1.Secret{}
		res, _, err := sr.Decode(nil, resultSecret)
		if err != nil {
			t.Fatalf("%v:\n%s", err, hex.Dump(obj.Bytes()))
		}
		resultSecret.Kind = "Secret"
		resultSecret.APIVersion = "v1"
		if !apiequality.Semantic.DeepEqual(v1secret, res) {
			t.Fatalf("objects did not match: %s", diff.ObjectGoPrintDiff(v1secret, res))
		}

		// write a watch event through the frame writer and read it back in
		// via the frame reader for this media type
		obj = &bytes.Buffer{}
		if err := embedded.Encode(v1secret, obj); err != nil {
			t.Fatal(err)
		}
		event := &metav1.WatchEvent{Type: string(watch.Added)}
		event.Object.Raw = obj.Bytes()
		obj = &bytes.Buffer{}
		if err := s.Encode(event, obj); err != nil {
			t.Fatal(err)
		}
		out = &bytes.Buffer{}
		w = framer.NewFrameWriter(out)
		if n, err := w.Write(obj.Bytes()); err != nil || n != len(obj.Bytes()) {
			t.Fatal(err)
		}
		sr = streaming.NewDecoder(framer.NewFrameReader(ioutil.NopCloser(out)), s)
		outEvent := &metav1.WatchEvent{}
		res, _, err = sr.Decode(nil, outEvent)
		if err != nil || outEvent.Type != string(watch.Added) {
			t.Fatalf("%v: %#v", err, outEvent)
		}
		if outEvent.Object.Object == nil && outEvent.Object.Raw != nil {
			outEvent.Object.Object, err = runtime.Decode(innerDecode, outEvent.Object.Raw)
			if err != nil {
				t.Fatalf("%v:\n%s", err, hex.Dump(outEvent.Object.Raw))
			}
		}

		if !apiequality.Semantic.DeepEqual(secret, outEvent.Object.Object) {
			t.Fatalf("%s: did not match after frame decoding: %s", info.MediaType, diff.ObjectGoPrintDiff(secret, outEvent.Object.Object))
		}
	}
}

const benchmarkSeed = 100

func benchmarkItems(b *testing.B) []v1.Pod {
	apiObjectFuzzer := apitesting.FuzzerFor(kapitesting.FuzzerFuncs(b, api.Codecs), rand.NewSource(benchmarkSeed))
	items := make([]v1.Pod, 10)
	for i := range items {
		var pod api.Pod
		apiObjectFuzzer.Fuzz(&pod)
		pod.Spec.InitContainers, pod.Status.InitContainerStatuses = nil, nil
		out, err := api.Scheme.ConvertToVersion(&pod, v1.SchemeGroupVersion)
		if err != nil {
			panic(err)
		}
		items[i] = *out.(*v1.Pod)
	}
	return items
}

// BenchmarkEncodeCodec measures the cost of performing a codec encode, which includes
// reflection (to clear APIVersion and Kind)
func BenchmarkEncodeCodec(b *testing.B) {
	items := benchmarkItems(b)
	width := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := runtime.Encode(testapi.Default.Codec(), &items[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkEncodeCodecFromInternal measures the cost of performing a codec encode,
// including conversions.
func BenchmarkEncodeCodecFromInternal(b *testing.B) {
	items := benchmarkItems(b)
	width := len(items)
	encodable := make([]api.Pod, width)
	for i := range items {
		if err := api.Scheme.Convert(&items[i], &encodable[i], nil); err != nil {
			b.Fatal(err)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := runtime.Encode(testapi.Default.Codec(), &encodable[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkEncodeJSONMarshal provides a baseline for regular JSON encode performance
func BenchmarkEncodeJSONMarshal(b *testing.B) {
	items := benchmarkItems(b)
	width := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := json.Marshal(&items[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

func BenchmarkDecodeCodec(b *testing.B) {
	codec := testapi.Default.Codec()
	items := benchmarkItems(b)
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(codec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := runtime.Decode(codec, encoded[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

func BenchmarkDecodeIntoExternalCodec(b *testing.B) {
	codec := testapi.Default.Codec()
	items := benchmarkItems(b)
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(codec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := runtime.DecodeInto(codec, encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

func BenchmarkDecodeIntoInternalCodec(b *testing.B) {
	codec := testapi.Default.Codec()
	items := benchmarkItems(b)
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(codec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := api.Pod{}
		if err := runtime.DecodeInto(codec, encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeJSON provides a baseline for regular JSON decode performance
func BenchmarkDecodeIntoJSON(b *testing.B) {
	codec := testapi.Default.Codec()
	items := benchmarkItems(b)
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(codec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := json.Unmarshal(encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeJSON provides a baseline for codecgen JSON decode performance
func BenchmarkDecodeIntoJSONCodecGen(b *testing.B) {
	kcodec := testapi.Default.Codec()
	items := benchmarkItems(b)
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := runtime.Encode(kcodec, &items[i])
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
	}
	handler := &codec.JsonHandle{}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := codec.NewDecoderBytes(encoded[i%width], handler).Decode(&obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

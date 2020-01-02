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

package testing

import (
	"bytes"
	"encoding/hex"
	gojson "encoding/json"
	"io/ioutil"
	"math/rand"
	"reflect"
	"testing"

	jsoniter "github.com/json-iterator/go"
	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/apitesting/roundtrip"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/apis/apps"
	k8s_apps_v1 "k8s.io/kubernetes/pkg/apis/apps/v1"
	api "k8s.io/kubernetes/pkg/apis/core"
	k8s_api_v1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"sigs.k8s.io/yaml"
)

// fuzzInternalObject fuzzes an arbitrary runtime object using the appropriate
// fuzzer registered with the apitesting package.
func fuzzInternalObject(t *testing.T, forVersion schema.GroupVersion, item runtime.Object, seed int64) runtime.Object {
	fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(seed), legacyscheme.Codecs).Fuzz(item)

	j, err := meta.TypeAccessor(item)
	if err != nil {
		t.Fatalf("Unexpected error %v for %#v", err, item)
	}
	j.SetKind("")
	j.SetAPIVersion("")

	return item
}

func ConvertV1ReplicaSetToAPIReplicationController(in *appsv1.ReplicaSet, out *api.ReplicationController, s conversion.Scope) error {
	intermediate1 := &apps.ReplicaSet{}
	if err := k8s_apps_v1.Convert_v1_ReplicaSet_To_apps_ReplicaSet(in, intermediate1, s); err != nil {
		return err
	}

	intermediate2 := &v1.ReplicationController{}
	if err := k8s_api_v1.Convert_apps_ReplicaSet_To_v1_ReplicationController(intermediate1, intermediate2, s); err != nil {
		return err
	}

	return k8s_api_v1.Convert_v1_ReplicationController_To_core_ReplicationController(intermediate2, out, s)
}

func TestSetControllerConversion(t *testing.T) {
	s := legacyscheme.Scheme
	if err := s.AddConversionFunc((*appsv1.ReplicaSet)(nil), (*api.ReplicationController)(nil), func(a, b interface{}, scope conversion.Scope) error {
		return ConvertV1ReplicaSetToAPIReplicationController(a.(*appsv1.ReplicaSet), b.(*api.ReplicationController), scope)
	}); err != nil {
		t.Fatal(err)
	}

	rs := &apps.ReplicaSet{}
	rc := &api.ReplicationController{}
	extGroup := schema.GroupVersion{Group: "apps", Version: "v1"}
	extCodec := legacyscheme.Codecs.LegacyCodec(extGroup)

	defaultGroup := schema.GroupVersion{Group: "", Version: "v1"}
	defaultCodec := legacyscheme.Codecs.LegacyCodec(defaultGroup)

	fuzzInternalObject(t, schema.GroupVersion{Group: "apps", Version: runtime.APIVersionInternal}, rs, rand.Int63())

	// explicitly set the selector to something that is convertible to old-style selectors
	// (since normally we'll fuzz the selectors with things that aren't convertible)
	rs.Spec.Selector = &metav1.LabelSelector{
		MatchLabels: map[string]string{
			"foo": "bar",
			"baz": "quux",
		},
	}

	t.Logf("rs._internal.apps -> rs.v1.apps")
	data, err := runtime.Encode(extCodec, rs)
	if err != nil {
		t.Fatalf("unexpected encoding error: %v", err)
	}

	decoder := legacyscheme.Codecs.DecoderToVersion(
		legacyscheme.Codecs.UniversalDeserializer(),
		runtime.NewMultiGroupVersioner(
			defaultGroup,
			schema.GroupKind{Group: defaultGroup.Group},
			schema.GroupKind{Group: extGroup.Group},
		),
	)

	t.Logf("rs.v1.apps -> rc._internal")
	if err := runtime.DecodeInto(decoder, data, rc); err != nil {
		t.Fatalf("unexpected decoding error: %v", err)
	}

	t.Logf("rc._internal -> rc.v1")
	data, err = runtime.Encode(defaultCodec, rc)
	if err != nil {
		t.Fatalf("unexpected encoding error: %v", err)
	}

	t.Logf("rc.v1 -> rs._internal.apps")
	if err := runtime.DecodeInto(decoder, data, rs); err != nil {
		t.Fatalf("unexpected decoding error: %v", err)
	}
}

// TestSpecificKind round-trips a single specific kind and is intended to help
// debug issues that arise while adding a new API type.
func TestSpecificKind(t *testing.T) {
	// Uncomment the following line to enable logging of which conversions
	// legacyscheme.Scheme.Log(t)
	internalGVK := schema.GroupVersionKind{Group: "apps", Version: runtime.APIVersionInternal, Kind: "DaemonSet"}

	seed := rand.Int63()
	fuzzer := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(seed), legacyscheme.Codecs)

	roundtrip.RoundTripSpecificKind(t, internalGVK, legacyscheme.Scheme, legacyscheme.Codecs, fuzzer, nil)
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
	// DeleteOptions, CreateOptions and UpdateOptions are only read in metav1
	"DeleteOptions",
	"CreateOptions",
	"UpdateOptions",
	"PatchOptions",
)

var commonKinds = []string{"Status", "ListOptions", "DeleteOptions", "ExportOptions", "GetOptions", "CreateOptions", "UpdateOptions", "PatchOptions"}

// TestCommonKindsRegistered verifies that all group/versions registered with
// the legacyscheme package have the common kinds.
func TestCommonKindsRegistered(t *testing.T) {
	gvs := map[schema.GroupVersion]bool{}
	for gvk := range legacyscheme.Scheme.AllKnownTypes() {
		if gvk.Version == runtime.APIVersionInternal {
			continue
		}
		gvs[gvk.GroupVersion()] = true
	}

	for _, kind := range commonKinds {
		for gv := range gvs {
			gvk := gv.WithKind(kind)
			obj, err := legacyscheme.Scheme.New(gvk)
			if err != nil {
				t.Error(err)
			}
			defaults := gv.WithKind("")
			var got *schema.GroupVersionKind
			if obj, got, err = legacyscheme.Codecs.LegacyCodec().Decode([]byte(`{"kind":"`+kind+`"}`), &defaults, obj); err != nil || gvk != *got {
				t.Errorf("expected %v: %v %v", gvk, got, err)
			}
			data, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(gv), obj)
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
// in all of the API groups registered for test in the legacyscheme package.
func TestRoundTripTypes(t *testing.T) {
	seed := rand.Int63()
	fuzzer := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(seed), legacyscheme.Codecs)
	nonRoundTrippableTypes := map[schema.GroupVersionKind]bool{}

	roundtrip.RoundTripTypes(t, legacyscheme.Scheme, legacyscheme.Codecs, fuzzer, nonRoundTrippableTypes)
}

// TestEncodePtr tests that a pointer to a golang type can be encoded and
// decoded without information loss or mutation.
func TestEncodePtr(t *testing.T) {
	grace := int64(30)
	enableServiceLinks := v1.DefaultEnableServiceLinks
	preemptNever := api.PreemptNever
	pod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Labels: map[string]string{"name": "foo"},
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,

			TerminationGracePeriodSeconds: &grace,

			SecurityContext:    &api.PodSecurityContext{},
			SchedulerName:      api.DefaultSchedulerName,
			EnableServiceLinks: &enableServiceLinks,
			PreemptionPolicy:   &preemptNever,
		},
	}
	obj := runtime.Object(pod)
	data, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), obj)
	obj2, err2 := runtime.Decode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), data)
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

func TestDecodeTimeStampWithoutQuotes(t *testing.T) {
	testYAML := []byte(`
apiVersion: v1
kind: Pod
metadata:
  creationTimestamp: 2018-08-30T14:10:58Z
  name: test
spec:
  containers: null
status: {}`)
	if obj, err := runtime.Decode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), testYAML); err != nil {
		t.Fatalf("unable to decode yaml: %v", err)
	} else {
		if obj2, ok := obj.(*api.Pod); !ok {
			t.Fatalf("Got wrong type")
		} else {
			if obj2.ObjectMeta.CreationTimestamp.UnixNano() != parseTimeOrDie("2018-08-30T14:10:58Z").UnixNano() {
				t.Fatalf("Time stamps do not match")
			}
		}
	}
}

// TestBadJSONRejection establishes that a JSON object without a kind or with
// an unknown kind will not be decoded without error.
func TestBadJSONRejection(t *testing.T) {
	badJSONMissingKind := []byte(`{ }`)
	if _, err := runtime.Decode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), badJSONMissingKind); err == nil {
		t.Errorf("Did not reject despite lack of kind field: %s", badJSONMissingKind)
	}
	badJSONUnknownType := []byte(`{"kind": "bar"}`)
	if _, err1 := runtime.Decode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), badJSONUnknownType); err1 == nil {
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
		unversionedJSON, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), obj)
		if err != nil {
			t.Errorf("%v: unexpected error: %v", obj, err)
			continue
		}

		// Make sure the versioned codec under test can decode
		versionDecodedObject, err := runtime.Decode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), unversionedJSON)
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
	f := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(benchmarkSeed), legacyscheme.Codecs)
	secret := &api.Secret{}
	f.Fuzz(secret)
	if secret.Data == nil {
		secret.Data = map[string][]byte{}
	}
	secret.Data["binary"] = []byte{0x00, 0x10, 0x30, 0x55, 0xff, 0x00}
	secret.Data["utf8"] = []byte("a string with \u0345 characters")
	secret.Data["long"] = bytes.Repeat([]byte{0x01, 0x02, 0x03, 0x00}, 1000)
	converted, _ := legacyscheme.Scheme.ConvertToVersion(secret, v1.SchemeGroupVersion)
	v1secret := converted.(*v1.Secret)
	for _, info := range legacyscheme.Codecs.SupportedMediaTypes() {
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
		innerDecode := legacyscheme.Codecs.DecoderToVersion(embedded, api.SchemeGroupVersion)

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
		_, _, err = sr.Decode(nil, outEvent)
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
	apiObjectFuzzer := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(benchmarkSeed), legacyscheme.Codecs)
	items := make([]v1.Pod, 10)
	for i := range items {
		var pod api.Pod
		apiObjectFuzzer.Fuzz(&pod)
		pod.Spec.InitContainers, pod.Status.InitContainerStatuses = nil, nil
		out, err := legacyscheme.Scheme.ConvertToVersion(&pod, v1.SchemeGroupVersion)
		if err != nil {
			panic(err)
		}
		items[i] = *out.(*v1.Pod)
	}
	return items
}

func benchmarkItemsList(b *testing.B, numItems int) v1.PodList {
	apiObjectFuzzer := fuzzer.FuzzerFor(FuzzerFuncs, rand.NewSource(benchmarkSeed), legacyscheme.Codecs)
	items := make([]v1.Pod, numItems)
	for i := range items {
		var pod api.Pod
		apiObjectFuzzer.Fuzz(&pod)
		pod.Spec.InitContainers, pod.Status.InitContainerStatuses = nil, nil
		out, err := legacyscheme.Scheme.ConvertToVersion(&pod, v1.SchemeGroupVersion)
		if err != nil {
			panic(err)
		}
		items[i] = *out.(*v1.Pod)
	}

	return v1.PodList{
		Items: items,
	}
}

// BenchmarkEncodeCodec measures the cost of performing a codec encode, which includes
// reflection (to clear APIVersion and Kind)
func BenchmarkEncodeCodec(b *testing.B) {
	items := benchmarkItems(b)
	width := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &items[i%width]); err != nil {
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
		if err := legacyscheme.Scheme.Convert(&items[i], &encodable[i], nil); err != nil {
			b.Fatal(err)
		}
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := runtime.Encode(legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), &encodable[i%width]); err != nil {
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
		if _, err := gojson.Marshal(&items[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

func BenchmarkDecodeCodec(b *testing.B) {
	codec := legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion)
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
	codec := legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion)
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
	codec := legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion)
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
	codec := legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion)
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
		if err := gojson.Unmarshal(encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeIntoJSONCodecGenConfigFast provides a baseline
// for JSON decode performance with jsoniter.ConfigFast
func BenchmarkDecodeIntoJSONCodecGenConfigFast(b *testing.B) {
	kcodec := legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion)
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

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := jsoniter.ConfigFastest.Unmarshal(encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeIntoJSONCodecGenConfigCompatibleWithStandardLibrary provides a
// baseline for JSON decode performance with
// jsoniter.ConfigCompatibleWithStandardLibrary, but with case sensitivity set
// to true
func BenchmarkDecodeIntoJSONCodecGenConfigCompatibleWithStandardLibrary(b *testing.B) {
	kcodec := legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion)
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

	b.ResetTimer()
	iter := json.CaseSensitiveJsonIterator()
	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := iter.Unmarshal(encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkEncodeYAMLMarshal provides a baseline for regular YAML encode performance
func BenchmarkEncodeYAMLMarshal(b *testing.B) {
	items := benchmarkItems(b)
	width := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := yaml.Marshal(&items[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeYAML provides a baseline for regular YAML decode performance
func BenchmarkDecodeIntoYAML(b *testing.B) {
	codec := legacyscheme.Codecs.LegacyCodec(v1.SchemeGroupVersion)
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
		if err := yaml.Unmarshal(encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

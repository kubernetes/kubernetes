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
	"fmt"
	"io/ioutil"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/davecgh/go-spew/spew"
	proto "github.com/golang/protobuf/proto"
	flag "github.com/spf13/pflag"
	"github.com/ugorji/go/codec"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/streaming"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/watch"
	"k8s.io/kubernetes/pkg/watch/versioned"
)

var fuzzIters = flag.Int("fuzz-iters", 20, "How many fuzzing iterations to do.")

var codecsToTest = []func(version unversioned.GroupVersion, item runtime.Object) (runtime.Codec, bool, error){
	func(version unversioned.GroupVersion, item runtime.Object) (runtime.Codec, bool, error) {
		c, err := testapi.GetCodecForObject(item)
		return c, true, err
	},
}

func fuzzInternalObject(t *testing.T, forVersion unversioned.GroupVersion, item runtime.Object, seed int64) runtime.Object {
	apitesting.FuzzerFor(t, forVersion, rand.NewSource(seed)).Fuzz(item)

	j, err := meta.TypeAccessor(item)
	if err != nil {
		t.Fatalf("Unexpected error %v for %#v", err, item)
	}
	j.SetKind("")
	j.SetAPIVersion("")

	return item
}

func dataAsString(data []byte) string {
	dataString := string(data)
	if !strings.HasPrefix(dataString, "{") {
		dataString = "\n" + hex.Dump(data)
		proto.NewBuffer(make([]byte, 0, 1024)).DebugPrint("decoded object", data)
	}
	return dataString
}

func roundTrip(t *testing.T, codec runtime.Codec, item runtime.Object) {
	printer := spew.ConfigState{DisableMethods: true}

	original := item
	copied, err := api.Scheme.DeepCopy(item)
	if err != nil {
		panic(fmt.Sprintf("unable to copy: %v", err))
	}
	item = copied.(runtime.Object)

	name := reflect.TypeOf(item).Elem().Name()
	data, err := runtime.Encode(codec, item)
	if err != nil {
		if runtime.IsNotRegisteredError(err) {
			t.Logf("%v: not registered: %v (%s)", name, err, printer.Sprintf("%#v", item))
		} else {
			t.Errorf("%v: %v (%s)", name, err, printer.Sprintf("%#v", item))
		}
		return
	}

	if !api.Semantic.DeepEqual(original, item) {
		t.Errorf("0: %v: encode altered the object, diff: %v", name, diff.ObjectReflectDiff(original, item))
		return
	}

	obj2, err := runtime.Decode(codec, data)
	if err != nil {
		t.Errorf("0: %v: %v\nCodec: %#v\nData: %s\nSource: %#v", name, err, codec, dataAsString(data), printer.Sprintf("%#v", item))
		panic("failed")
	}
	if !api.Semantic.DeepEqual(original, obj2) {
		t.Errorf("\n1: %v: diff: %v\nCodec: %#v\nSource:\n\n%#v\n\nEncoded:\n\n%s\n\nFinal:\n\n%#v", name, diff.ObjectReflectDiff(item, obj2), codec, printer.Sprintf("%#v", item), dataAsString(data), printer.Sprintf("%#v", obj2))
		return
	}

	obj3 := reflect.New(reflect.TypeOf(item).Elem()).Interface().(runtime.Object)
	if err := runtime.DecodeInto(codec, data, obj3); err != nil {
		t.Errorf("2: %v: %v", name, err)
		return
	}
	if !api.Semantic.DeepEqual(item, obj3) {
		t.Errorf("3: %v: diff: %v\nCodec: %#v", name, diff.ObjectReflectDiff(item, obj3), codec)
		return
	}
}

// roundTripSame verifies the same source object is tested in all API versions.
func roundTripSame(t *testing.T, group testapi.TestGroup, item runtime.Object, except ...string) {
	set := sets.NewString(except...)
	seed := rand.Int63()
	fuzzInternalObject(t, group.InternalGroupVersion(), item, seed)

	version := *group.GroupVersion()
	codecs := []runtime.Codec{}
	for _, fn := range codecsToTest {
		codec, ok, err := fn(version, item)
		if err != nil {
			t.Errorf("unable to get codec: %v", err)
			return
		}
		if !ok {
			continue
		}
		codecs = append(codecs, codec)
	}

	if !set.Has(version.String()) {
		fuzzInternalObject(t, version, item, seed)
		for _, codec := range codecs {
			roundTrip(t, codec, item)
		}
	}
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
			unversioned.GroupKind{Group: defaultGroup.GroupVersion().Group},
			unversioned.GroupKind{Group: extGroup.GroupVersion().Group},
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

// For debugging problems
func TestSpecificKind(t *testing.T) {
	kind := "DaemonSet"
	for i := 0; i < *fuzzIters; i++ {
		doRoundTripTest(testapi.Groups["extensions"], kind, t)
		if t.Failed() {
			break
		}
	}
}

func TestList(t *testing.T) {
	kind := "List"
	item, err := api.Scheme.New(api.SchemeGroupVersion.WithKind(kind))
	if err != nil {
		t.Errorf("Couldn't make a %v? %v", kind, err)
		return
	}
	roundTripSame(t, testapi.Default, item)
}

var nonRoundTrippableTypes = sets.NewString(
	"ExportOptions",
	// WatchEvent does not include kind and version and can only be deserialized
	// implicitly (if the caller expects the specific object). The watch call defines
	// the schema by content type, rather than via kind/version included in each
	// object.
	"WatchEvent",
)

var commonKinds = []string{"Status", "ListOptions", "DeleteOptions", "ExportOptions"}

// verify all external group/versions have the common kinds registered.
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
			if _, got, err := api.Codecs.LegacyCodec().Decode([]byte(`{"kind":"`+kind+`"}`), &defaults, nil); err != nil || gvk != *got {
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

var nonInternalRoundTrippableTypes = sets.NewString("List", "ListOptions", "ExportOptions")
var nonRoundTrippableTypesByVersion = map[string][]string{}

func TestRoundTripTypes(t *testing.T) {
	for groupKey, group := range testapi.Groups {
		for kind := range group.InternalTypes() {
			t.Logf("working on %v in %v", kind, groupKey)
			if nonRoundTrippableTypes.Has(kind) {
				continue
			}
			// Try a few times, since runTest uses random values.
			for i := 0; i < *fuzzIters; i++ {
				doRoundTripTest(group, kind, t)
				if t.Failed() {
					break
				}
			}
		}
	}
}

func doRoundTripTest(group testapi.TestGroup, kind string, t *testing.T) {
	item, err := api.Scheme.New(group.InternalGroupVersion().WithKind(kind))
	if err != nil {
		t.Fatalf("Couldn't make a %v? %v", kind, err)
	}
	if _, err := meta.TypeAccessor(item); err != nil {
		t.Fatalf("%q is not a TypeMeta and cannot be tested - add it to nonRoundTrippableTypes: %v", kind, err)
	}
	if api.Scheme.Recognizes(group.GroupVersion().WithKind(kind)) {
		roundTripSame(t, group, item, nonRoundTrippableTypesByVersion[kind]...)
	}
	if !nonInternalRoundTrippableTypes.Has(kind) && api.Scheme.Recognizes(group.GroupVersion().WithKind(kind)) {
		roundTrip(t, group.Codec(), fuzzInternalObject(t, group.InternalGroupVersion(), item, rand.Int63()))
	}
}

func TestEncode_Ptr(t *testing.T) {
	grace := int64(30)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			Labels: map[string]string{"name": "foo"},
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSClusterFirst,

			TerminationGracePeriodSeconds: &grace,

			SecurityContext: &api.PodSecurityContext{},
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
	if !api.Semantic.DeepEqual(obj2, pod) {
		t.Errorf("\nExpected:\n\n %#v,\n\nGot:\n\n %#vDiff: %v\n\n", pod, obj2, diff.ObjectDiff(obj2, pod))

	}
}

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

func TestUnversionedTypes(t *testing.T) {
	testcases := []runtime.Object{
		&unversioned.Status{Status: "Failure", Message: "something went wrong"},
		&unversioned.APIVersions{Versions: []string{"A", "B", "C"}},
		&unversioned.APIGroupList{Groups: []unversioned.APIGroup{{Name: "mygroup"}}},
		&unversioned.APIGroup{Name: "mygroup"},
		&unversioned.APIResourceList{GroupVersion: "mygroup/myversion"},
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

func TestObjectWatchFraming(t *testing.T) {
	f := apitesting.FuzzerFor(nil, api.SchemeGroupVersion, rand.NewSource(benchmarkSeed))
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
		if !api.Semantic.DeepEqual(v1secret, res) {
			t.Fatalf("objects did not match: %s", diff.ObjectGoPrintDiff(v1secret, res))
		}

		// write a watch event through and back out
		obj = &bytes.Buffer{}
		if err := embedded.Encode(v1secret, obj); err != nil {
			t.Fatal(err)
		}
		event := &versioned.Event{Type: string(watch.Added)}
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
		outEvent := &versioned.Event{}
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

		if !api.Semantic.DeepEqual(secret, outEvent.Object.Object) {
			t.Fatalf("%s: did not match after frame decoding: %s", info.MediaType, diff.ObjectGoPrintDiff(secret, outEvent.Object.Object))
		}
	}
}

const benchmarkSeed = 100

func benchmarkItems() []v1.Pod {
	apiObjectFuzzer := apitesting.FuzzerFor(nil, api.SchemeGroupVersion, rand.NewSource(benchmarkSeed))
	items := make([]v1.Pod, 2)
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
	items := benchmarkItems()
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
	items := benchmarkItems()
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
	items := benchmarkItems()
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
	items := benchmarkItems()
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
	items := benchmarkItems()
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
	items := benchmarkItems()
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
	items := benchmarkItems()
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
	items := benchmarkItems()
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

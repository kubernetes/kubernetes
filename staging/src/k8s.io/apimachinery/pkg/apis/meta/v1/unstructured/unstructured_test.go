/*
Copyright 2018 The Kubernetes Authors.

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

package unstructured_test

import (
	"bytes"
	"math/big"
	"math/rand"
	"os"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/stretchr/testify/assert"
	"sigs.k8s.io/randfill"

	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/equality"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	cborserializer "k8s.io/apimachinery/pkg/runtime/serializer/cbor"
	jsonserializer "k8s.io/apimachinery/pkg/runtime/serializer/json"
)

func TestNilUnstructuredContent(t *testing.T) {
	var u unstructured.Unstructured
	uCopy := u.DeepCopy()
	content := u.UnstructuredContent()
	expContent := make(map[string]interface{})
	assert.EqualValues(t, expContent, content)
	assert.Equal(t, uCopy, &u)
}

// TestUnstructuredMetadataRoundTrip checks that metadata accessors
// correctly set the metadata for unstructured objects.
// First, it fuzzes an empty ObjectMeta and sets this value as the metadata for an unstructured object.
// Next, it uses metadata accessor methods to set these fuzzed values to another unstructured object.
// Finally, it checks that both the unstructured objects are equal.
func TestUnstructuredMetadataRoundTrip(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	seed := rand.Int63()
	fuzzer := fuzzer.FuzzerFor(metafuzzer.Funcs, rand.NewSource(seed), codecs)

	N := 1000
	for i := 0; i < N; i++ {
		u := &unstructured.Unstructured{Object: map[string]interface{}{}}
		uCopy := u.DeepCopy()
		metadata := &metav1.ObjectMeta{}
		fuzzer.Fill(metadata)

		if err := setObjectMeta(u, metadata); err != nil {
			t.Fatalf("unexpected error setting fuzzed ObjectMeta: %v", err)
		}
		setObjectMetaUsingAccessors(u, uCopy)

		if !equality.Semantic.DeepEqual(u, uCopy) {
			t.Errorf("diff: %v", cmp.Diff(u, uCopy))
		}
	}
}

// TestUnstructuredMetadataOmitempty checks that ObjectMeta omitempty
// semantics are enforced for unstructured objects.
// The fuzzing test above should catch these cases but this is here just to be safe.
// Example: the metadata.clusterName field has the omitempty json tag
// so if it is set to it's zero value (""), it should be removed from the metadata map.
func TestUnstructuredMetadataOmitempty(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	seed := rand.Int63()
	fuzzer := fuzzer.FuzzerFor(metafuzzer.Funcs, rand.NewSource(seed), codecs)

	// fuzz to make sure we don't miss any function calls below
	u := &unstructured.Unstructured{Object: map[string]interface{}{}}
	metadata := &metav1.ObjectMeta{}
	fuzzer.Fill(metadata)
	if err := setObjectMeta(u, metadata); err != nil {
		t.Fatalf("unexpected error setting fuzzed ObjectMeta: %v", err)
	}

	// set zero values for all fields in metadata explicitly
	// to check that omitempty fields having zero values are never set
	u.SetName("")
	u.SetGenerateName("")
	u.SetNamespace("")
	u.SetSelfLink("")
	u.SetUID("")
	u.SetResourceVersion("")
	u.SetGeneration(0)
	u.SetCreationTimestamp(metav1.Time{})
	u.SetDeletionTimestamp(nil)
	u.SetDeletionGracePeriodSeconds(nil)
	u.SetLabels(nil)
	u.SetAnnotations(nil)
	u.SetOwnerReferences(nil)
	u.SetFinalizers(nil)
	u.SetManagedFields(nil)

	gotMetadata, _, err := unstructured.NestedFieldNoCopy(u.UnstructuredContent(), "metadata")
	if err != nil {
		t.Error(err)
	}
	emptyMetadata := make(map[string]interface{})

	if !reflect.DeepEqual(gotMetadata, emptyMetadata) {
		t.Errorf("expected %v, got %v", emptyMetadata, gotMetadata)
	}
}

// TestRoundTripJSONCBORUnstructured performs fuzz testing for roundtrip for
// unstructured object between JSON and CBOR
func TestRoundTripJSONCBORUnstructured(t *testing.T) {
	roundtripType[*unstructured.Unstructured](t)
}

// TestRoundTripJSONCBORUnstructuredList performs fuzz testing for roundtrip for
// unstructuredList object between JSON and CBOR
func TestRoundTripJSONCBORUnstructuredList(t *testing.T) {
	roundtripType[*unstructured.UnstructuredList](t)
}

func setObjectMeta(u *unstructured.Unstructured, objectMeta *metav1.ObjectMeta) error {
	if objectMeta == nil {
		unstructured.RemoveNestedField(u.UnstructuredContent(), "metadata")
		return nil
	}
	metadata, err := runtime.DefaultUnstructuredConverter.ToUnstructured(objectMeta)
	if err != nil {
		return err
	}
	u.UnstructuredContent()["metadata"] = metadata
	return nil
}

func setObjectMetaUsingAccessors(u, uCopy *unstructured.Unstructured) {
	uCopy.SetName(u.GetName())
	uCopy.SetGenerateName(u.GetGenerateName())
	uCopy.SetNamespace(u.GetNamespace())
	uCopy.SetSelfLink(u.GetSelfLink())
	uCopy.SetUID(u.GetUID())
	uCopy.SetResourceVersion(u.GetResourceVersion())
	uCopy.SetGeneration(u.GetGeneration())
	uCopy.SetCreationTimestamp(u.GetCreationTimestamp())
	uCopy.SetDeletionTimestamp(u.GetDeletionTimestamp())
	uCopy.SetDeletionGracePeriodSeconds(u.GetDeletionGracePeriodSeconds())
	uCopy.SetLabels(u.GetLabels())
	uCopy.SetAnnotations(u.GetAnnotations())
	uCopy.SetOwnerReferences(u.GetOwnerReferences())
	uCopy.SetFinalizers(u.GetFinalizers())
	uCopy.SetManagedFields(u.GetManagedFields())
}

// roundtripType performs fuzz testing for roundtrip conversion for
// unstructured or unstructuredList object between two formats (A and B) in forward
// and backward directions
// Original and final unstructured/list are compared along with all intermediate ones
func roundtripType[U runtime.Unstructured](t *testing.T) {
	scheme := runtime.NewScheme()
	fuzzer := fuzzer.FuzzerFor(fuzzer.MergeFuzzerFuncs(metafuzzer.Funcs, unstructuredFuzzerFuncs), rand.NewSource(getSeed(t)), serializer.NewCodecFactory(scheme))

	jS := jsonserializer.NewSerializerWithOptions(jsonserializer.DefaultMetaFactory, scheme, scheme, jsonserializer.SerializerOptions{})
	cS := cborserializer.NewSerializer(scheme, scheme)

	for i := 0; i < 50; i++ {
		original := reflect.New(reflect.TypeFor[U]().Elem()).Interface().(runtime.Unstructured)
		fuzzer.Fill(original)
		// unstructured -> JSON > unstructured > CBOR -> unstructured -> JSON -> unstructured
		roundtrip(t, original, jS, cS)
		// unstructured -> CBOR > unstructured > JSON -> unstructured -> CBOR -> unstructured
		roundtrip(t, original, cS, jS)
	}
}

// roundtrip tests that an Unstructured object roundtrips faithfully along the
// sequence Unstructured -> A -> Unstructured -> B -> Unstructured -> A -> Unstructured,
// given serializers for two encodings A and B. The final object and both intermediate
// objects must all be equal to the original.
func roundtrip(t *testing.T, original runtime.Unstructured, a, b runtime.Serializer) {
	var buf bytes.Buffer

	buf.Reset()
	// (original) Unstructured -> A
	if err := a.Encode(original, &buf); err != nil {
		t.Fatalf("error encoding original unstructured to A: %v", err)
	}
	// A -> intermediate unstructured
	uA := reflect.New(reflect.TypeOf(original).Elem()).Interface().(runtime.Object)
	uA, _, err := a.Decode(buf.Bytes(), nil, uA)
	if err != nil {
		t.Fatalf("error decoding A to unstructured: %v", err)
	}

	// Compare original unstructured vs intermediate unstructured
	tmp, ok := uA.(runtime.Unstructured)
	if !ok {
		t.Fatalf("unexpected type %T for unstructured", tmp)
	}
	if !unstructuredEqual(t, original, uA.(runtime.Unstructured)) {
		t.Fatalf("original unstructured differed from unstructured via A: %v", cmp.Diff(original, uA))
	}

	buf.Reset()
	// intermediate unstructured -> B
	if err := b.Encode(uA, &buf); err != nil {
		t.Fatalf("error encoding unstructured to B: %v", err)
	}
	// B -> intermediate unstructured
	uB := reflect.New(reflect.TypeOf(original).Elem()).Interface().(runtime.Object)
	uB, _, err = b.Decode(buf.Bytes(), nil, uB)
	if err != nil {
		t.Fatalf("error decoding B to unstructured: %v", err)
	}

	// compare original vs intermediate unstructured
	tmp, ok = uB.(runtime.Unstructured)
	if !ok {
		t.Fatalf("unexpected type %T for unstructured", tmp)
	}
	if !unstructuredEqual(t, original, uB.(runtime.Unstructured)) {
		t.Fatalf("unstructured via A differed from unstructured via B: %v", cmp.Diff(original, uB))
	}

	// intermediate unstructured -> A
	buf.Reset()
	if err := a.Encode(uB, &buf); err != nil {
		t.Fatalf("error encoding unstructured to A: %v", err)
	}
	// A -> final unstructured
	final := reflect.New(reflect.TypeOf(original).Elem()).Interface().(runtime.Object)
	final, _, err = a.Decode(buf.Bytes(), nil, final)
	if err != nil {
		t.Fatalf("error decoding A to unstructured: %v", err)
	}

	// Compare original unstructured vs final unstructured
	tmp, ok = final.(runtime.Unstructured)
	if !ok {
		t.Fatalf("unexpected type %T for unstructured", tmp)
	}
	if !unstructuredEqual(t, original, final.(runtime.Unstructured)) {
		t.Errorf("object changed during unstructured->A->unstructured->B->unstructured roundtrip, diff: %s", cmp.Diff(original, final))
	}
}

func getSeed(t *testing.T) int64 {
	seed := int64(time.Now().Nanosecond())
	if override := os.Getenv("TEST_RAND_SEED"); len(override) > 0 {
		overrideSeed, err := strconv.ParseInt(override, 10, 64)
		if err != nil {
			t.Fatal(err)
		}
		seed = overrideSeed
		t.Logf("using overridden seed: %d", seed)
	} else {
		t.Logf("seed (override with TEST_RAND_SEED if desired): %d", seed)
	}
	return seed
}

const (
	maxUnstructuredDepth  = 64
	maxUnstructuredFanOut = 5
)

func unstructuredFuzzerFuncs(codecs serializer.CodecFactory) []interface{} {
	return []interface{}{
		func(u *unstructured.Unstructured, c randfill.Continue) {
			obj := make(map[string]interface{})
			obj["apiVersion"] = generateValidAPIVersionString(c)
			obj["kind"] = generateNonEmptyString(c)
			for j := c.Intn(maxUnstructuredFanOut); j >= 0; j-- {
				obj[c.String(0)] = generateRandomTypeValue(maxUnstructuredDepth, c)
			}
			u.Object = obj
		},
		func(ul *unstructured.UnstructuredList, c randfill.Continue) {
			obj := make(map[string]interface{})
			obj["apiVersion"] = generateValidAPIVersionString(c)
			obj["kind"] = generateNonEmptyString(c)
			for j := c.Intn(maxUnstructuredFanOut); j >= 0; j-- {
				obj[c.String(0)] = generateRandomTypeValue(maxUnstructuredDepth, c)
			}
			for j := c.Intn(maxUnstructuredFanOut); j >= 0; j-- {
				var item = unstructured.Unstructured{}
				c.Fill(&item)
				ul.Items = append(ul.Items, item)
			}
			ul.Object = obj
		},
	}
}

func generateNonEmptyString(c randfill.Continue) string {
	temp := c.String(0)
	for len(temp) == 0 {
		temp = c.String(0)
	}
	return temp
}

// generateNonEmptyNoSlashString generates a non-empty string without any slashes
func generateNonEmptyNoSlashString(c randfill.Continue) string {
	temp := strings.ReplaceAll(generateNonEmptyString(c), "/", "")
	for len(temp) == 0 {
		temp = strings.ReplaceAll(generateNonEmptyString(c), "/", "")
	}
	return temp
}

// generateValidAPIVersionString generates valid apiVersion string with formats:
// <string>/<string> or <string>
func generateValidAPIVersionString(c randfill.Continue) string {
	if c.Bool() {
		return generateNonEmptyNoSlashString(c) + "/" + generateNonEmptyNoSlashString(c)
	} else {
		return generateNonEmptyNoSlashString(c)
	}
}

// generateRandomTypeValue generates fuzzed valid JSON data types:
// 1. numbers (float64, int64)
// 2. string (utf-8 encodings)
// 3. boolean
// 4. array ([]interface{})
// 5. object (map[string]interface{})
// 6. null
// Decoding into unstructured can only produce a nil interface{} value or the
// concrete types map[string]interface{}, []interface{}, int64, float64, string, and bool
// If a value of other types is put into an unstructured, it will roundtrip
// to one of the above list of supported types. For example, if Time type is used,
// it will be encoded into a RFC 3339 format string such as "2001-02-03T12:34:56Z"
// and when decoding into Unstructured, there is no information to indicate
// that this string was originally produced by encoding a metav1.Time.
// All external-versioned builtin types are exercised through RoundtripToUnstructured
// in apitesting package. Types like metav1.Time are implicitly being exercised
// because they appear as fields in those types.
func generateRandomTypeValue(depth int, c randfill.Continue) interface{} {
	t := c.Rand.Intn(120)
	// If the max depth for unstructured is reached, only add non-recursive types
	// which is 20+ in range
	if depth == 0 {
		t = 20 + c.Rand.Intn(120-20)
	}

	switch {
	case t < 10:
		item := make([]interface{}, c.Intn(maxUnstructuredFanOut))
		for k := range item {
			item[k] = generateRandomTypeValue(depth-1, c)
		}
		return item
	case t < 20:
		item := map[string]interface{}{}
		for j := c.Intn(maxUnstructuredFanOut); j >= 0; j-- {
			item[c.String(0)] = generateRandomTypeValue(depth-1, c)
		}
		return item
	case t < 40:
		// Only valid UTF-8 encodings
		var item string
		c.Fill(&item)
		return item
	case t < 60:
		var item int64
		c.Fill(&item)
		return item
	case t < 80:
		var item bool
		c.Fill(&item)
		return item
	case t < 100:
		return c.Rand.NormFloat64()
	case t < 120:
		return nil
	default:
		panic("invalid case")
	}
}

func unstructuredEqual(t *testing.T, a, b runtime.Unstructured) bool {
	return anyEqual(t, a.UnstructuredContent(), b.UnstructuredContent())
}

// numberEqual asserts equality of two numbers which one is int64 and one is float64
// In JSON, a non-decimal float64 is converted to int64 automatically in case the
// float64 fits into int64 range. Otherwise, the non-decimal float64 remains a float.
// As a result, this func does an int64 to float64 conversion using math/big package
// to ensure the conversion is lossless before comparison.
func numberEqual(a int64, b float64) bool {
	// Ensure roundtrip int64 to float64 conversion is lossless
	f, accuracy := big.NewInt(a).Float64()
	if accuracy == big.Exact {
		// Distinction between int64 and float64 is not preserved during JSON roundtrip for all numbers.
		return f == b
	}
	return false
}

func anyEqual(t *testing.T, a, b interface{}) bool {
	switch b.(type) {
	case nil, bool, string, int64, float64, []interface{}, map[string]interface{}:
	default:
		t.Fatalf("unexpected value %v of type %T", b, b)
	}

	switch ac := a.(type) {
	case nil, bool, string:
		return ac == b
	case int64:
		if bc, ok := b.(float64); ok {
			return numberEqual(ac, bc)
		}
		return ac == b
	case float64:
		if bc, ok := b.(int64); ok {
			return numberEqual(bc, ac)
		}
		return ac == b
	case []interface{}:
		bc, ok := b.([]interface{})
		if !ok {
			return false
		}
		if len(ac) != len(bc) {
			return false
		}
		for i, aa := range ac {
			if !anyEqual(t, aa, bc[i]) {
				return false
			}
		}
		return true
	case map[string]interface{}:
		bc, ok := b.(map[string]interface{})
		if !ok {
			return false
		}
		if len(ac) != len(bc) {
			return false
		}
		for k, aa := range ac {
			bb, ok := bc[k]
			if !ok {
				return false
			}
			if !anyEqual(t, aa, bb) {
				return false
			}
		}
		return true
	default:
		t.Fatalf("unexpected value %v of type %T", a, a)
	}
	return true
}

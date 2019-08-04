/*
Copyright 2019 The Kubernetes Authors.

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

package roundtrip

import (
	"bytes"
	"io/ioutil"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	fuzz "github.com/google/gofuzz"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	genericfuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	"k8s.io/apimachinery/pkg/util/sets"
)

// CompatibilityTestOptions holds configuration for running a compatibility test using in-memory objects
// and serialized files on disk representing the current code and serialized data from previous versions.
//
// Example use: `NewCompatibilityTestOptions(scheme).Complete(t).Run(t)`
type CompatibilityTestOptions struct {
	// Scheme is used to create new objects for fuzzing, decoding, and for constructing serializers.
	// Required.
	Scheme *runtime.Scheme

	// TestDataDir points to a directory containing compatibility test data.
	// Complete() populates this with "testdata" if unset.
	TestDataDir string

	// TestDataDirCurrentVersion points to a directory containing compatibility test data for the current version.
	// Complete() populates this with "<TestDataDir>/HEAD" if unset.
	// Within this directory, `<group>.<version>.<kind>.[json|yaml|pb]` files are required to exist, and are:
	// * verified to match serialized FuzzedObjects[GVK]
	// * verified to decode without error
	// * verified to round-trip byte-for-byte when re-encoded
	// * verified to be semantically equal when decoded into memory
	TestDataDirCurrentVersion string

	// TestDataDirsPreviousVersions is a list of directories containing compatibility test data for previous versions.
	// Complete() populates this with "<TestDataDir>/v*" directories if nil.
	// Within these directories, `<group>.<version>.<kind>.[json|yaml|pb]` files are optional. If present, they are:
	// * verified to decode without error
	// * verified to round-trip byte-for-byte when re-encoded (or to match a `<group>.<version>.<kind>.[json|yaml|pb].after_roundtrip.[json|yaml|pb]` file if it exists)
	// * verified to be semantically equal when decoded into memory
	TestDataDirsPreviousVersions []string

	// Kinds is a list of fully qualified kinds to test.
	// Complete() populates this with Scheme.AllKnownTypes() if unset.
	Kinds []schema.GroupVersionKind

	// FuzzedObjects is an optional set of fuzzed objects to use for verifying HEAD fixtures.
	// Complete() populates this with the result of CompatibilityTestObject(Kinds[*], Scheme, FuzzFuncs) for any missing kinds.
	// Objects must be deterministically fuzzed and identical on every invocation.
	FuzzedObjects map[schema.GroupVersionKind]runtime.Object

	// FuzzFuncs is an optional set of custom fuzzing functions to use to construct FuzzedObjects.
	// They *must* not use any random source other than the passed-in fuzzer.
	FuzzFuncs []interface{}

	JSON  runtime.Serializer
	YAML  runtime.Serializer
	Proto runtime.Serializer
}

func NewCompatibilityTestOptions(scheme *runtime.Scheme) *CompatibilityTestOptions {
	return &CompatibilityTestOptions{Scheme: scheme}
}

// coreKinds includes kinds that typically only need to be tested in a single API group
var coreKinds = sets.NewString(
	"CreateOptions", "UpdateOptions", "PatchOptions", "DeleteOptions",
	"GetOptions", "ListOptions", "ExportOptions",
	"WatchEvent",
)

func (c *CompatibilityTestOptions) Complete(t *testing.T) *CompatibilityTestOptions {
	t.Helper()

	// Verify scheme
	if c.Scheme == nil {
		t.Fatal("scheme is required")
	}

	// Populate testdata dirs
	if c.TestDataDir == "" {
		c.TestDataDir = "testdata"
	}
	if c.TestDataDirCurrentVersion == "" {
		c.TestDataDirCurrentVersion = filepath.Join(c.TestDataDir, "HEAD")
	}
	if c.TestDataDirsPreviousVersions == nil {
		dirs, err := filepath.Glob(filepath.Join(c.TestDataDir, "v*"))
		if err != nil {
			t.Fatal(err)
		}
		sort.Strings(dirs)
		c.TestDataDirsPreviousVersions = dirs
	}

	// Populate kinds
	if len(c.Kinds) == 0 {
		gvks := []schema.GroupVersionKind{}
		for gvk := range c.Scheme.AllKnownTypes() {
			if gvk.Version == "" || gvk.Version == runtime.APIVersionInternal {
				// only test external types
				continue
			}
			if strings.HasSuffix(gvk.Kind, "List") {
				// omit list types
				continue
			}
			if gvk.Group != "" && coreKinds.Has(gvk.Kind) {
				// only test options types in the core API group
				continue
			}
			gvks = append(gvks, gvk)
		}
		c.Kinds = gvks
	}

	// Sort kinds to get deterministic test order
	sort.Slice(c.Kinds, func(i, j int) bool {
		if c.Kinds[i].Group != c.Kinds[j].Group {
			return c.Kinds[i].Group < c.Kinds[j].Group
		}
		if c.Kinds[i].Version != c.Kinds[j].Version {
			return c.Kinds[i].Version < c.Kinds[j].Version
		}
		if c.Kinds[i].Kind != c.Kinds[j].Kind {
			return c.Kinds[i].Kind < c.Kinds[j].Kind
		}
		return false
	})

	// Fuzz any missing objects
	if c.FuzzedObjects == nil {
		c.FuzzedObjects = map[schema.GroupVersionKind]runtime.Object{}
	}
	for _, gvk := range c.Kinds {
		if _, ok := c.FuzzedObjects[gvk]; ok {
			continue
		}
		obj, err := CompatibilityTestObject(c.Scheme, gvk, c.FuzzFuncs)
		if err != nil {
			t.Fatal(err)
		}
		c.FuzzedObjects[gvk] = obj
	}

	if c.JSON == nil {
		c.JSON = json.NewSerializer(json.DefaultMetaFactory, c.Scheme, c.Scheme, true)
	}
	if c.YAML == nil {
		c.YAML = json.NewYAMLSerializer(json.DefaultMetaFactory, c.Scheme, c.Scheme)
	}
	if c.Proto == nil {
		c.Proto = protobuf.NewSerializer(c.Scheme, c.Scheme)
	}

	return c
}

// CompatibilityTestObject returns a deterministically fuzzed object for the specified GVK
func CompatibilityTestObject(scheme *runtime.Scheme, gvk schema.GroupVersionKind, fuzzFuncs []interface{}) (runtime.Object, error) {
	// Construct the object
	obj, err := scheme.New(gvk)
	if err != nil {
		return nil, err
	}

	// Fuzz it
	CompatibilityTestFuzzer(scheme, fuzzFuncs).Fuzz(obj)

	// Set the kind and apiVersion
	if typeAcc, err := apimeta.TypeAccessor(obj); err != nil {
		return nil, err
	} else {
		typeAcc.SetKind(gvk.Kind)
		typeAcc.SetAPIVersion(gvk.GroupVersion().String())
	}

	return obj, nil
}

// CompatibilityTestFuzzer returns a fuzzer for the given scheme:
// - fixed seed (deterministic output that lets us generate the same fixtures on every run)
// - 0 nil chance (populate all fields)
// - 1 numelements (populate and bound all lists)
// - 20 max depth (don't recurse infinitely)
// - meta fuzzing functions added
// - custom fuzzing functions to make strings and managedFields more readable in fixtures
func CompatibilityTestFuzzer(scheme *runtime.Scheme, fuzzFuncs []interface{}) *fuzz.Fuzzer {
	fuzzer := fuzz.NewWithSeed(0).NilChance(0).NumElements(1, 1).MaxDepth(20)
	fuzzer = fuzzer.Funcs(genericfuzzer.Funcs(serializer.NewCodecFactory(scheme))...)
	fuzzString := 1
	fuzzer.Funcs(
		// avoid crazy strings
		func(s *string, c fuzz.Continue) {
			fuzzString++
			*s = strconv.Itoa(fuzzString)
		},
		// limit managed fields to two levels
		func(f *[]metav1.ManagedFieldsEntry, c fuzz.Continue) {
			field := metav1.ManagedFieldsEntry{}
			c.Fuzz(&field)
			if field.Fields != nil {
				field.Fields.Raw = []byte("{}")
			}
			*f = []metav1.ManagedFieldsEntry{field}
		},
		func(r *runtime.RawExtension, c fuzz.Continue) {
			// generate a raw object in normalized form
			// TODO: test non-normalized round-tripping... YAMLToJSON normalizes and makes exact comparisons fail
			r.Raw = []byte(`{"apiVersion":"example.com/v1","kind":"CustomType","spec":{"replicas":1},"status":{"available":1}}`)
		},
	)
	fuzzer.Funcs(fuzzFuncs...)
	return fuzzer
}

func (c *CompatibilityTestOptions) Run(t *testing.T) {
	for _, gvk := range c.Kinds {
		t.Run(makeName(gvk), func(t *testing.T) {

			t.Run("HEAD", func(t *testing.T) {
				c.runCurrentVersionTest(t, gvk)
			})

			for _, previousVersionDir := range c.TestDataDirsPreviousVersions {
				t.Run(filepath.Base(previousVersionDir), func(t *testing.T) {
					c.runPreviousVersionTest(t, gvk, previousVersionDir)
				})
			}

		})
	}
}

func (c *CompatibilityTestOptions) runCurrentVersionTest(t *testing.T, gvk schema.GroupVersionKind) {
	expectedObject := c.FuzzedObjects[gvk]
	expectedJSON, expectedYAML, expectedProto := c.encode(t, expectedObject)

	actualJSON, actualYAML, actualProto, err := read(c.TestDataDirCurrentVersion, gvk, "")
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}

	needsUpdate := false
	if os.IsNotExist(err) {
		t.Errorf("current version compatibility files did not exist: %v", err)
		needsUpdate = true
	} else {
		if !bytes.Equal(expectedJSON, actualJSON) {
			t.Errorf("json differs")
			t.Log(cmp.Diff(string(expectedJSON), string(actualJSON)))
			needsUpdate = true
		}

		if !bytes.Equal(expectedYAML, actualYAML) {
			t.Errorf("yaml differs")
			t.Log(cmp.Diff(string(expectedYAML), string(actualYAML)))
			needsUpdate = true
		}

		if !bytes.Equal(expectedProto, actualProto) {
			t.Errorf("proto differs")
			needsUpdate = true
			t.Log(cmp.Diff(dumpProto(t, expectedProto[4:]), dumpProto(t, actualProto[4:])))
			// t.Logf("json (for locating the offending field based on surrounding data): %s", string(expectedJSON))
		}
	}

	if needsUpdate {
		const updateEnvVar = "UPDATE_COMPATIBILITY_FIXTURE_DATA"
		if os.Getenv(updateEnvVar) == "true" {
			writeFile(t, c.TestDataDirCurrentVersion, gvk, "", "json", expectedJSON)
			writeFile(t, c.TestDataDirCurrentVersion, gvk, "", "yaml", expectedYAML)
			writeFile(t, c.TestDataDirCurrentVersion, gvk, "", "pb", expectedProto)
			t.Logf("wrote expected compatibility data... verify, commit, and rerun tests")
		} else {
			t.Logf("if the diff is expected because of a new type or a new field, re-run with %s=true to update the compatibility data", updateEnvVar)
		}
		return
	}

	emptyObj, err := c.Scheme.New(gvk)
	if err != nil {
		t.Fatal(err)
	}
	{
		jsonDecoded := emptyObj.DeepCopyObject()
		jsonDecoded, _, err = c.JSON.Decode(actualJSON, &gvk, jsonDecoded)
		if err != nil {
			t.Error(err)
		} else if !apiequality.Semantic.DeepEqual(expectedObject, jsonDecoded) {
			t.Errorf("expected and decoded json objects differed:\n%s", cmp.Diff(expectedObject, jsonDecoded))
		}
	}
	{
		yamlDecoded := emptyObj.DeepCopyObject()
		yamlDecoded, _, err = c.YAML.Decode(actualYAML, &gvk, yamlDecoded)
		if err != nil {
			t.Error(err)
		} else if !apiequality.Semantic.DeepEqual(expectedObject, yamlDecoded) {
			t.Errorf("expected and decoded yaml objects differed:\n%s", cmp.Diff(expectedObject, yamlDecoded))
		}
	}
	{
		protoDecoded := emptyObj.DeepCopyObject()
		protoDecoded, _, err = c.Proto.Decode(actualProto, &gvk, protoDecoded)
		if err != nil {
			t.Error(err)
		} else if !apiequality.Semantic.DeepEqual(expectedObject, protoDecoded) {
			t.Errorf("expected and decoded proto objects differed:\n%s", cmp.Diff(expectedObject, protoDecoded))
		}
	}
}

func (c *CompatibilityTestOptions) encode(t *testing.T, obj runtime.Object) (json, yaml, proto []byte) {
	jsonBytes := bytes.NewBuffer(nil)
	if err := c.JSON.Encode(obj, jsonBytes); err != nil {
		t.Fatalf("error encoding json: %v", err)
	}
	yamlBytes := bytes.NewBuffer(nil)
	if err := c.YAML.Encode(obj, yamlBytes); err != nil {
		t.Fatalf("error encoding yaml: %v", err)
	}
	protoBytes := bytes.NewBuffer(nil)
	if err := c.Proto.Encode(obj, protoBytes); err != nil {
		t.Fatalf("error encoding proto: %v", err)
	}
	return jsonBytes.Bytes(), yamlBytes.Bytes(), protoBytes.Bytes()
}

func read(dir string, gvk schema.GroupVersionKind, suffix string) (json, yaml, proto []byte, err error) {
	actualJSON, jsonErr := ioutil.ReadFile(filepath.Join(dir, makeName(gvk)+suffix+".json"))
	actualYAML, yamlErr := ioutil.ReadFile(filepath.Join(dir, makeName(gvk)+suffix+".yaml"))
	actualProto, protoErr := ioutil.ReadFile(filepath.Join(dir, makeName(gvk)+suffix+".pb"))
	if jsonErr != nil {
		return actualJSON, actualYAML, actualProto, jsonErr
	}
	if yamlErr != nil {
		return actualJSON, actualYAML, actualProto, yamlErr
	}
	if protoErr != nil {
		return actualJSON, actualYAML, actualProto, protoErr
	}
	return actualJSON, actualYAML, actualProto, nil
}

func writeFile(t *testing.T, dir string, gvk schema.GroupVersionKind, suffix, extension string, data []byte) {
	if err := os.MkdirAll(dir, os.FileMode(0755)); err != nil {
		t.Fatal("error making directory", err)
	}
	if err := ioutil.WriteFile(filepath.Join(dir, makeName(gvk)+suffix+"."+extension), data, os.FileMode(0644)); err != nil {
		t.Fatalf("error writing %s: %v", extension, err)
	}
}

func (c *CompatibilityTestOptions) runPreviousVersionTest(t *testing.T, gvk schema.GroupVersionKind, previousVersionDir string) {
	jsonBeforeRoundTrip, yamlBeforeRoundTrip, protoBeforeRoundTrip, err := read(previousVersionDir, gvk, "")
	if os.IsNotExist(err) || (len(jsonBeforeRoundTrip) == 0 && len(yamlBeforeRoundTrip) == 0 && len(protoBeforeRoundTrip) == 0) {
		t.SkipNow()
		return
	}
	if err != nil {
		t.Fatal(err)
	}

	emptyObj, err := c.Scheme.New(gvk)
	if err != nil {
		t.Fatal(err)
	}

	jsonDecoded := emptyObj.DeepCopyObject()
	jsonDecoded, _, err = c.JSON.Decode(jsonBeforeRoundTrip, &gvk, jsonDecoded)
	if err != nil {
		t.Fatal(err)
	}
	jsonBytes := bytes.NewBuffer(nil)
	if err := c.JSON.Encode(jsonDecoded, jsonBytes); err != nil {
		t.Fatalf("error encoding json: %v", err)
	}
	jsonAfterRoundTrip := jsonBytes.Bytes()

	yamlDecoded := emptyObj.DeepCopyObject()
	yamlDecoded, _, err = c.YAML.Decode(yamlBeforeRoundTrip, &gvk, yamlDecoded)
	if err != nil {
		t.Fatal(err)
	} else if !apiequality.Semantic.DeepEqual(jsonDecoded, yamlDecoded) {
		t.Errorf("decoded json and yaml objects differ:\n%s", cmp.Diff(jsonDecoded, yamlDecoded))
	}
	yamlBytes := bytes.NewBuffer(nil)
	if err := c.YAML.Encode(yamlDecoded, yamlBytes); err != nil {
		t.Fatalf("error encoding yaml: %v", err)
	}
	yamlAfterRoundTrip := yamlBytes.Bytes()

	protoDecoded := emptyObj.DeepCopyObject()
	protoDecoded, _, err = c.Proto.Decode(protoBeforeRoundTrip, &gvk, protoDecoded)
	if err != nil {
		t.Fatal(err)
	} else if !apiequality.Semantic.DeepEqual(jsonDecoded, protoDecoded) {
		t.Errorf("decoded json and proto objects differ:\n%s", cmp.Diff(jsonDecoded, protoDecoded))
	}
	protoBytes := bytes.NewBuffer(nil)
	if err := c.Proto.Encode(protoDecoded, protoBytes); err != nil {
		t.Fatalf("error encoding proto: %v", err)
	}
	protoAfterRoundTrip := protoBytes.Bytes()

	expectedJSONAfterRoundTrip, expectedYAMLAfterRoundTrip, expectedProtoAfterRoundTrip, _ := read(previousVersionDir, gvk, ".after_roundtrip")
	if len(expectedJSONAfterRoundTrip) == 0 {
		expectedJSONAfterRoundTrip = jsonBeforeRoundTrip
	}
	if len(expectedYAMLAfterRoundTrip) == 0 {
		expectedYAMLAfterRoundTrip = yamlBeforeRoundTrip
	}
	if len(expectedProtoAfterRoundTrip) == 0 {
		expectedProtoAfterRoundTrip = protoBeforeRoundTrip
	}

	jsonNeedsUpdate := false
	yamlNeedsUpdate := false
	protoNeedsUpdate := false

	if !bytes.Equal(expectedJSONAfterRoundTrip, jsonAfterRoundTrip) {
		t.Errorf("json differs")
		t.Log(cmp.Diff(string(expectedJSONAfterRoundTrip), string(jsonAfterRoundTrip)))
		jsonNeedsUpdate = true
	}

	if !bytes.Equal(expectedYAMLAfterRoundTrip, yamlAfterRoundTrip) {
		t.Errorf("yaml differs")
		t.Log(cmp.Diff(string(expectedYAMLAfterRoundTrip), string(yamlAfterRoundTrip)))
		yamlNeedsUpdate = true
	}

	if !bytes.Equal(expectedProtoAfterRoundTrip, protoAfterRoundTrip) {
		t.Errorf("proto differs")
		protoNeedsUpdate = true
		t.Log(cmp.Diff(dumpProto(t, expectedProtoAfterRoundTrip[4:]), dumpProto(t, protoAfterRoundTrip[4:])))
		// t.Logf("json (for locating the offending field based on surrounding data): %s", string(expectedJSON))
	}

	if jsonNeedsUpdate || yamlNeedsUpdate || protoNeedsUpdate {
		const updateEnvVar = "UPDATE_COMPATIBILITY_FIXTURE_DATA"
		if os.Getenv(updateEnvVar) == "true" {
			if jsonNeedsUpdate {
				writeFile(t, previousVersionDir, gvk, ".after_roundtrip", "json", jsonAfterRoundTrip)
			}
			if yamlNeedsUpdate {
				writeFile(t, previousVersionDir, gvk, ".after_roundtrip", "yaml", yamlAfterRoundTrip)
			}
			if protoNeedsUpdate {
				writeFile(t, previousVersionDir, gvk, ".after_roundtrip", "pb", protoAfterRoundTrip)
			}
			t.Logf("wrote expected compatibility data... verify, commit, and rerun tests")
		} else {
			t.Logf("if the diff is expected because of a new type or a new field, re-run with %s=true to update the compatibility data", updateEnvVar)
		}
		return
	}
}

func makeName(gvk schema.GroupVersionKind) string {
	g := gvk.Group
	if g == "" {
		g = "core"
	}
	return g + "." + gvk.Version + "." + gvk.Kind
}

func dumpProto(t *testing.T, data []byte) string {
	t.Helper()
	protoc, err := exec.LookPath("protoc")
	if err != nil {
		t.Log(err)
		return ""
	}
	cmd := exec.Command(protoc, "--decode_raw")
	cmd.Stdin = bytes.NewBuffer(data)
	d, err := cmd.CombinedOutput()
	if err != nil {
		t.Log(err)
		return ""
	}
	return string(d)
}

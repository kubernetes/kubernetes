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

package kubeletplugin

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/roundtrip"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/dynamic-resource-allocation/api/metadata"
	"k8s.io/dynamic-resource-allocation/devicemetadata"
)

const updateCompatibilityFixtureData = "UPDATE_COMPATIBILITY_FIXTURE_DATA"

func TestDeviceMetadataCompatibility(t *testing.T) {
	expected := compatibilityTestMetadata(t)
	versions := registeredMetadataVersions()
	if len(versions) == 0 {
		t.Fatal("no external metadata API versions are registered")
	}

	expectedData := encodeCompatibilityMetadata(t, expected, versions)
	testDataDir := filepath.Join("..", "api", "metadata", "testdata")
	headFixture := filepath.Join(testDataDir, "HEAD", "metadata.json")
	if !matchCompatibilityFixture(t, headFixture, expectedData) {
		return
	}

	t.Run("HEAD", func(t *testing.T) {
		requireSingleMetadataFixture(t, filepath.Dir(headFixture))
		testMetadataFixture(t, headFixture, expected)
	})

	previousVersionDirs, err := filepath.Glob(filepath.Join(testDataDir, "v*"))
	if err != nil {
		t.Fatal(err)
	}
	if len(previousVersionDirs) == 0 {
		t.Fatal("no previous-version compatibility fixtures found")
	}
	for _, dir := range previousVersionDirs {
		t.Run(filepath.Base(dir), func(t *testing.T) {
			requireSingleMetadataFixture(t, dir)
			testMetadataFixture(t, filepath.Join(dir, "metadata.json"), nil)
		})
	}
}

func requireSingleMetadataFixture(t *testing.T, dir string) {
	t.Helper()

	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].Name() != "metadata.json" {
		t.Fatalf("%s must contain only metadata.json", dir)
	}
}

func compatibilityTestMetadata(t *testing.T) *metadata.DeviceMetadata {
	t.Helper()

	fillFuncs := map[reflect.Type]roundtrip.FillFunc{
		reflect.TypeFor[*metav1.TypeMeta](): func(_ string, _ int, obj interface{}) {
			*obj.(*metav1.TypeMeta) = metav1.TypeMeta{}
		},
		reflect.TypeFor[*metav1.FieldsV1](): func(_ string, _ int, obj interface{}) {
			obj.(*metav1.FieldsV1).SetRawString("{}")
		},
		reflect.TypeFor[*metav1.Time](): func(_ string, i int, obj interface{}) {
			obj.(*metav1.Time).Time = time.Date(2000+i, 1, 1, 1, 1, 1, 0, time.UTC)
		},
		reflect.TypeFor[*metav1.MicroTime](): func(_ string, i int, obj interface{}) {
			obj.(*metav1.MicroTime).Time = time.Date(2000+i, 1, 1, 1, 1, 1, i*int(time.Microsecond), time.UTC)
		},
	}
	obj, err := roundtrip.CompatibilityTestObject(
		metadataScheme,
		metadata.SchemeGroupVersion.WithKind("DeviceMetadata"),
		fillFuncs,
	)
	if err != nil {
		t.Fatal(err)
	}
	// setting the fields is not strictly necessary, but it helps to generate
	// human readable test data that semantically makes sense.
	dm := obj.(*metadata.DeviceMetadata)
	dm.TypeMeta = metav1.TypeMeta{}
	dm.Name = "claim"
	dm.Namespace = "default"
	dm.UID = "claim-uid"
	dm.Requests[0].Name = "request"
	dm.Requests[0].Devices[0].Driver = "gpu.example.com"
	dm.Requests[0].Devices[0].Pool = "worker-0"
	dm.Requests[0].Devices[0].Name = "gpu-0"
	for _, attribute := range dm.Requests[0].Devices[0].Attributes {
		model := "LATEST-GPU-MODEL"
		attribute.StringValue = &model
		dm.Requests[0].Devices[0].Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
			"model": attribute,
		}
		break
	}
	return dm
}

func registeredMetadataVersions() []schema.GroupVersion {
	versionSet := map[schema.GroupVersion]struct{}{}
	for gvk := range metadataScheme.AllKnownTypes() {
		if gvk.Group != metadata.GroupName || gvk.Version == runtime.APIVersionInternal || gvk.Kind != "DeviceMetadata" {
			continue
		}
		versionSet[gvk.GroupVersion()] = struct{}{}
	}

	versions := make([]schema.GroupVersion, 0, len(versionSet))
	for gv := range versionSet {
		versions = append(versions, gv)
	}
	sort.Slice(versions, func(i, j int) bool {
		return version.CompareKubeAwareVersionStrings(versions[i].Version, versions[j].Version) > 0
	})
	return versions
}

func encodeCompatibilityMetadata(t *testing.T, dm *metadata.DeviceMetadata, versions []schema.GroupVersion) []byte {
	t.Helper()

	writer, err := newMetadataWriter(
		"gpu.example.com",
		"",
		"",
		versions,
		defaultMetadataFileOperations(MetadataFileOperations{}),
	)
	if err != nil {
		t.Fatal(err)
	}
	data, err := writer.encodeMetadataStream(dm)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func matchCompatibilityFixture(t *testing.T, path string, expected []byte) bool {
	t.Helper()

	actual, err := os.ReadFile(path)
	if err == nil && bytes.Equal(actual, expected) {
		return true
	}
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}

	if err == nil {
		t.Errorf("%s differs (-want +got):\n%s", path, cmp.Diff(string(actual), string(expected)))
	} else {
		t.Errorf("%s does not exist", path)
	}
	if os.Getenv(updateCompatibilityFixtureData) == "true" {
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, expected, 0644); err != nil {
			t.Fatal(err)
		}
		t.Logf("wrote %s; verify, commit, and rerun tests", path)
	} else {
		t.Logf("rerun with %s=true to update compatibility data", updateCompatibilityFixtureData)
	}
	return false
}

func testMetadataFixture(t *testing.T, path string, expected *metadata.DeviceMetadata) {
	t.Helper()

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	// Decode without validation because metadata.managedFields[0].operation: "operationValue" is not valid.
	var decoded metadata.DeviceMetadata
	if err := devicemetadata.DecodeMetadataFromStream(json.NewDecoder(bytes.NewReader(data)), &decoded, devicemetadata.DecodeMetadataWithValidation(false)); err != nil {
		t.Fatalf("decode %s: %v", path, err)
	}
	if expected != nil && !apiequality.Semantic.DeepEqual(expected, &decoded) {
		t.Errorf("decoded metadata differs (-want +got):\n%s", cmp.Diff(expected, &decoded))
	}

	versions := metadataVersionsInStream(t, data)
	roundTripped := encodeCompatibilityMetadata(t, &decoded, versions)
	if !bytes.Equal(data, roundTripped) {
		t.Errorf("metadata changed after round trip (-want +got):\n%s", cmp.Diff(string(data), string(roundTripped)))
	}
}

func metadataVersionsInStream(t *testing.T, data []byte) []schema.GroupVersion {
	t.Helper()

	var versions []schema.GroupVersion
	decoder := json.NewDecoder(bytes.NewReader(data))
	for decoder.More() {
		var typeMeta metav1.TypeMeta
		if err := decoder.Decode(&typeMeta); err != nil {
			t.Fatal(err)
		}
		if typeMeta.Kind != "DeviceMetadata" {
			t.Fatalf("unexpected metadata kind %q", typeMeta.Kind)
		}
		gv, err := schema.ParseGroupVersion(typeMeta.APIVersion)
		if err != nil {
			t.Fatal(err)
		}
		versions = append(versions, gv)
	}
	if len(versions) == 0 {
		t.Fatal("metadata stream is empty")
	}
	return versions
}

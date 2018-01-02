package metadata

import (
	"io/ioutil"
	"os"
	"runtime"
	"testing"

	"github.com/docker/docker/layer"
)

func TestV1IDService(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "v1-id-service-test")
	if err != nil {
		t.Fatalf("could not create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	metadataStore, err := NewFSMetadataStore(tmpDir, runtime.GOOS)
	if err != nil {
		t.Fatalf("could not create metadata store: %v", err)
	}
	v1IDService := NewV1IDService(metadataStore)

	testVectors := []struct {
		registry string
		v1ID     string
		layerID  layer.DiffID
	}{
		{
			registry: "registry1",
			v1ID:     "f0cd5ca10b07f35512fc2f1cbf9a6cefbdb5cba70ac6b0c9e5988f4497f71937",
			layerID:  layer.DiffID("sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4"),
		},
		{
			registry: "registry2",
			v1ID:     "9e3447ca24cb96d86ebd5960cb34d1299b07e0a0e03801d90b9969a2c187dd6e",
			layerID:  layer.DiffID("sha256:86e0e091d0da6bde2456dbb48306f3956bbeb2eae1b5b9a43045843f69fe4aaa"),
		},
		{
			registry: "registry1",
			v1ID:     "9e3447ca24cb96d86ebd5960cb34d1299b07e0a0e03801d90b9969a2c187dd6e",
			layerID:  layer.DiffID("sha256:03f4658f8b782e12230c1783426bd3bacce651ce582a4ffb6fbbfa2079428ecb"),
		},
	}

	// Set some associations
	for _, vec := range testVectors {
		err := v1IDService.Set(vec.v1ID, vec.registry, vec.layerID)
		if err != nil {
			t.Fatalf("error calling Set: %v", err)
		}
	}

	// Check the correct values are read back
	for _, vec := range testVectors {
		layerID, err := v1IDService.Get(vec.v1ID, vec.registry)
		if err != nil {
			t.Fatalf("error calling Get: %v", err)
		}
		if layerID != vec.layerID {
			t.Fatal("Get returned incorrect layer ID")
		}
	}

	// Test Get on a nonexistent entry
	_, err = v1IDService.Get("82379823067823853223359023576437723560923756b03560378f4497753917", "registry1")
	if err == nil {
		t.Fatal("expected error looking up nonexistent entry")
	}

	// Overwrite one of the entries and read it back
	err = v1IDService.Set(testVectors[0].v1ID, testVectors[0].registry, testVectors[1].layerID)
	if err != nil {
		t.Fatalf("error calling Set: %v", err)
	}
	layerID, err := v1IDService.Get(testVectors[0].v1ID, testVectors[0].registry)
	if err != nil {
		t.Fatalf("error calling Get: %v", err)
	}
	if layerID != testVectors[1].layerID {
		t.Fatal("Get returned incorrect layer ID")
	}
}

package metadata

import (
	"encoding/hex"
	"io/ioutil"
	"math/rand"
	"os"
	"reflect"
	"runtime"
	"testing"

	"github.com/docker/docker/layer"
	"github.com/opencontainers/go-digest"
)

func TestV2MetadataService(t *testing.T) {
	tmpDir, err := ioutil.TempDir("", "blobsum-storage-service-test")
	if err != nil {
		t.Fatalf("could not create temp dir: %v", err)
	}
	defer os.RemoveAll(tmpDir)

	metadataStore, err := NewFSMetadataStore(tmpDir, runtime.GOOS)
	if err != nil {
		t.Fatalf("could not create metadata store: %v", err)
	}
	V2MetadataService := NewV2MetadataService(metadataStore)

	tooManyBlobSums := make([]V2Metadata, 100)
	for i := range tooManyBlobSums {
		randDigest := randomDigest()
		tooManyBlobSums[i] = V2Metadata{Digest: randDigest}
	}

	testVectors := []struct {
		diffID   layer.DiffID
		metadata []V2Metadata
	}{
		{
			diffID: layer.DiffID("sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4"),
			metadata: []V2Metadata{
				{Digest: digest.Digest("sha256:f0cd5ca10b07f35512fc2f1cbf9a6cefbdb5cba70ac6b0c9e5988f4497f71937")},
			},
		},
		{
			diffID: layer.DiffID("sha256:86e0e091d0da6bde2456dbb48306f3956bbeb2eae1b5b9a43045843f69fe4aaa"),
			metadata: []V2Metadata{
				{Digest: digest.Digest("sha256:f0cd5ca10b07f35512fc2f1cbf9a6cefbdb5cba70ac6b0c9e5988f4497f71937")},
				{Digest: digest.Digest("sha256:9e3447ca24cb96d86ebd5960cb34d1299b07e0a0e03801d90b9969a2c187dd6e")},
			},
		},
		{
			diffID:   layer.DiffID("sha256:03f4658f8b782e12230c1783426bd3bacce651ce582a4ffb6fbbfa2079428ecb"),
			metadata: tooManyBlobSums,
		},
	}

	// Set some associations
	for _, vec := range testVectors {
		for _, blobsum := range vec.metadata {
			err := V2MetadataService.Add(vec.diffID, blobsum)
			if err != nil {
				t.Fatalf("error calling Set: %v", err)
			}
		}
	}

	// Check the correct values are read back
	for _, vec := range testVectors {
		metadata, err := V2MetadataService.GetMetadata(vec.diffID)
		if err != nil {
			t.Fatalf("error calling Get: %v", err)
		}
		expectedMetadataEntries := len(vec.metadata)
		if expectedMetadataEntries > 50 {
			expectedMetadataEntries = 50
		}
		if !reflect.DeepEqual(metadata, vec.metadata[len(vec.metadata)-expectedMetadataEntries:len(vec.metadata)]) {
			t.Fatal("Get returned incorrect layer ID")
		}
	}

	// Test GetMetadata on a nonexistent entry
	_, err = V2MetadataService.GetMetadata(layer.DiffID("sha256:82379823067823853223359023576437723560923756b03560378f4497753917"))
	if err == nil {
		t.Fatal("expected error looking up nonexistent entry")
	}

	// Test GetDiffID on a nonexistent entry
	_, err = V2MetadataService.GetDiffID(digest.Digest("sha256:82379823067823853223359023576437723560923756b03560378f4497753917"))
	if err == nil {
		t.Fatal("expected error looking up nonexistent entry")
	}

	// Overwrite one of the entries and read it back
	err = V2MetadataService.Add(testVectors[1].diffID, testVectors[0].metadata[0])
	if err != nil {
		t.Fatalf("error calling Add: %v", err)
	}
	diffID, err := V2MetadataService.GetDiffID(testVectors[0].metadata[0].Digest)
	if err != nil {
		t.Fatalf("error calling GetDiffID: %v", err)
	}
	if diffID != testVectors[1].diffID {
		t.Fatal("GetDiffID returned incorrect diffID")
	}
}

func randomDigest() digest.Digest {
	b := [32]byte{}
	for i := 0; i < len(b); i++ {
		b[i] = byte(rand.Intn(256))
	}
	d := hex.EncodeToString(b[:])
	return digest.Digest("sha256:" + d)
}

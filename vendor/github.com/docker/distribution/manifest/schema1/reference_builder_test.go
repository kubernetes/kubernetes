package schema1

import (
	"testing"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/manifest"
	"github.com/docker/distribution/reference"
	"github.com/docker/libtrust"
	"github.com/opencontainers/go-digest"
)

func makeSignedManifest(t *testing.T, pk libtrust.PrivateKey, refs []Reference) *SignedManifest {
	u := &Manifest{
		Versioned: manifest.Versioned{
			SchemaVersion: 1,
		},
		Name:         "foo/bar",
		Tag:          "latest",
		Architecture: "amd64",
	}

	for i := len(refs) - 1; i >= 0; i-- {
		u.FSLayers = append(u.FSLayers, FSLayer{
			BlobSum: refs[i].Digest,
		})
		u.History = append(u.History, History{
			V1Compatibility: refs[i].History.V1Compatibility,
		})
	}

	signedManifest, err := Sign(u, pk)
	if err != nil {
		t.Fatalf("unexpected error signing manifest: %v", err)
	}
	return signedManifest
}

func TestReferenceBuilder(t *testing.T) {
	pk, err := libtrust.GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("unexpected error generating private key: %v", err)
	}

	r1 := Reference{
		Digest:  "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
		Size:    1,
		History: History{V1Compatibility: "{\"a\" : 1 }"},
	}
	r2 := Reference{
		Digest:  "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
		Size:    2,
		History: History{V1Compatibility: "{\"\a\" : 2 }"},
	}

	handCrafted := makeSignedManifest(t, pk, []Reference{r1, r2})

	ref, err := reference.WithName(handCrafted.Manifest.Name)
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}
	ref, err = reference.WithTag(ref, handCrafted.Manifest.Tag)
	if err != nil {
		t.Fatalf("could not add tag: %v", err)
	}

	b := NewReferenceManifestBuilder(pk, ref, handCrafted.Manifest.Architecture)
	_, err = b.Build(context.Background())
	if err == nil {
		t.Fatal("Expected error building zero length manifest")
	}

	err = b.AppendReference(r1)
	if err != nil {
		t.Fatal(err)
	}

	err = b.AppendReference(r2)
	if err != nil {
		t.Fatal(err)
	}

	refs := b.References()
	if len(refs) != 2 {
		t.Fatalf("Unexpected reference count : %d != %d", 2, len(refs))
	}

	// Ensure ordering
	if refs[0].Digest != r2.Digest {
		t.Fatalf("Unexpected reference : %v", refs[0])
	}

	m, err := b.Build(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	built, ok := m.(*SignedManifest)
	if !ok {
		t.Fatalf("unexpected type from Build() : %T", built)
	}

	d1 := digest.FromBytes(built.Canonical)
	d2 := digest.FromBytes(handCrafted.Canonical)
	if d1 != d2 {
		t.Errorf("mismatching canonical JSON")
	}
}

package schema1

import (
	"bytes"
	"encoding/json"
	"reflect"
	"testing"

	"github.com/docker/libtrust"
)

type testEnv struct {
	name, tag     string
	invalidSigned *SignedManifest
	signed        *SignedManifest
	pk            libtrust.PrivateKey
}

func TestManifestMarshaling(t *testing.T) {
	env := genEnv(t)

	// Check that the all field is the same as json.MarshalIndent with these
	// parameters.
	p, err := json.MarshalIndent(env.signed, "", "   ")
	if err != nil {
		t.Fatalf("error marshaling manifest: %v", err)
	}

	if !bytes.Equal(p, env.signed.all) {
		t.Fatalf("manifest bytes not equal: %q != %q", string(env.signed.all), string(p))
	}
}

func TestManifestUnmarshaling(t *testing.T) {
	env := genEnv(t)

	var signed SignedManifest
	if err := json.Unmarshal(env.signed.all, &signed); err != nil {
		t.Fatalf("error unmarshaling signed manifest: %v", err)
	}

	if !reflect.DeepEqual(&signed, env.signed) {
		t.Fatalf("manifests are different after unmarshaling: %v != %v", signed, env.signed)
	}

}

func TestManifestVerification(t *testing.T) {
	env := genEnv(t)

	publicKeys, err := Verify(env.signed)
	if err != nil {
		t.Fatalf("error verifying manifest: %v", err)
	}

	if len(publicKeys) == 0 {
		t.Fatalf("no public keys found in signature")
	}

	var found bool
	publicKey := env.pk.PublicKey()
	// ensure that one of the extracted public keys matches the private key.
	for _, candidate := range publicKeys {
		if candidate.KeyID() == publicKey.KeyID() {
			found = true
			break
		}
	}

	if !found {
		t.Fatalf("expected public key, %v, not found in verified keys: %v", publicKey, publicKeys)
	}

	// Check that an invalid manifest fails verification
	_, err = Verify(env.invalidSigned)
	if err != nil {
		t.Fatalf("Invalid manifest should not pass Verify()")
	}
}

func genEnv(t *testing.T) *testEnv {
	pk, err := libtrust.GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("error generating test key: %v", err)
	}

	name, tag := "foo/bar", "test"

	invalid := Manifest{
		Versioned: SchemaVersion,
		Name:      name,
		Tag:       tag,
		FSLayers: []FSLayer{
			{
				BlobSum: "asdf",
			},
			{
				BlobSum: "qwer",
			},
		},
	}

	valid := Manifest{
		Versioned: SchemaVersion,
		Name:      name,
		Tag:       tag,
		FSLayers: []FSLayer{
			{
				BlobSum: "asdf",
			},
		},
		History: []History{
			{
				V1Compatibility: "",
			},
		},
	}

	sm, err := Sign(&valid, pk)
	if err != nil {
		t.Fatalf("error signing manifest: %v", err)
	}

	invalidSigned, err := Sign(&invalid, pk)
	if err != nil {
		t.Fatalf("error signing manifest: %v", err)
	}

	return &testEnv{
		name:          name,
		tag:           tag,
		invalidSigned: invalidSigned,
		signed:        sm,
		pk:            pk,
	}
}

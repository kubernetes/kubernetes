package libtrust

import (
	"errors"
	"io/ioutil"
	"os"
	"testing"
)

func makeTempFile(t *testing.T, prefix string) (filename string) {
	file, err := ioutil.TempFile("", prefix)
	if err != nil {
		t.Fatal(err)
	}

	filename = file.Name()
	file.Close()

	return
}

func TestKeyFiles(t *testing.T) {
	key, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	testKeyFiles(t, key)

	key, err = GenerateRSA2048PrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	testKeyFiles(t, key)
}

func testKeyFiles(t *testing.T, key PrivateKey) {
	var err error

	privateKeyFilename := makeTempFile(t, "private_key")
	privateKeyFilenamePEM := privateKeyFilename + ".pem"
	privateKeyFilenameJWK := privateKeyFilename + ".jwk"

	publicKeyFilename := makeTempFile(t, "public_key")
	publicKeyFilenamePEM := publicKeyFilename + ".pem"
	publicKeyFilenameJWK := publicKeyFilename + ".jwk"

	if err = SaveKey(privateKeyFilenamePEM, key); err != nil {
		t.Fatal(err)
	}

	if err = SaveKey(privateKeyFilenameJWK, key); err != nil {
		t.Fatal(err)
	}

	if err = SavePublicKey(publicKeyFilenamePEM, key.PublicKey()); err != nil {
		t.Fatal(err)
	}

	if err = SavePublicKey(publicKeyFilenameJWK, key.PublicKey()); err != nil {
		t.Fatal(err)
	}

	loadedPEMKey, err := LoadKeyFile(privateKeyFilenamePEM)
	if err != nil {
		t.Fatal(err)
	}

	loadedJWKKey, err := LoadKeyFile(privateKeyFilenameJWK)
	if err != nil {
		t.Fatal(err)
	}

	loadedPEMPublicKey, err := LoadPublicKeyFile(publicKeyFilenamePEM)
	if err != nil {
		t.Fatal(err)
	}

	loadedJWKPublicKey, err := LoadPublicKeyFile(publicKeyFilenameJWK)
	if err != nil {
		t.Fatal(err)
	}

	if key.KeyID() != loadedPEMKey.KeyID() {
		t.Fatal(errors.New("key IDs do not match"))
	}

	if key.KeyID() != loadedJWKKey.KeyID() {
		t.Fatal(errors.New("key IDs do not match"))
	}

	if key.KeyID() != loadedPEMPublicKey.KeyID() {
		t.Fatal(errors.New("key IDs do not match"))
	}

	if key.KeyID() != loadedJWKPublicKey.KeyID() {
		t.Fatal(errors.New("key IDs do not match"))
	}

	os.Remove(privateKeyFilename)
	os.Remove(privateKeyFilenamePEM)
	os.Remove(privateKeyFilenameJWK)
	os.Remove(publicKeyFilename)
	os.Remove(publicKeyFilenamePEM)
	os.Remove(publicKeyFilenameJWK)
}

func TestTrustedHostKeysFile(t *testing.T) {
	trustedHostKeysFilename := makeTempFile(t, "trusted_host_keys")
	trustedHostKeysFilenamePEM := trustedHostKeysFilename + ".pem"
	trustedHostKeysFilenameJWK := trustedHostKeysFilename + ".json"

	testTrustedHostKeysFile(t, trustedHostKeysFilenamePEM)
	testTrustedHostKeysFile(t, trustedHostKeysFilenameJWK)

	os.Remove(trustedHostKeysFilename)
	os.Remove(trustedHostKeysFilenamePEM)
	os.Remove(trustedHostKeysFilenameJWK)
}

func testTrustedHostKeysFile(t *testing.T, trustedHostKeysFilename string) {
	hostAddress1 := "docker.example.com:2376"
	hostKey1, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	hostKey1.AddExtendedField("hosts", []string{hostAddress1})
	err = AddKeySetFile(trustedHostKeysFilename, hostKey1.PublicKey())
	if err != nil {
		t.Fatal(err)
	}

	trustedHostKeysMapping, err := LoadKeySetFile(trustedHostKeysFilename)
	if err != nil {
		t.Fatal(err)
	}

	for addr, hostKey := range trustedHostKeysMapping {
		t.Logf("Host Address: %d\n", addr)
		t.Logf("Host Key: %s\n\n", hostKey)
	}

	hostAddress2 := "192.168.59.103:2376"
	hostKey2, err := GenerateRSA2048PrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	hostKey2.AddExtendedField("hosts", hostAddress2)
	err = AddKeySetFile(trustedHostKeysFilename, hostKey2.PublicKey())
	if err != nil {
		t.Fatal(err)
	}

	trustedHostKeysMapping, err = LoadKeySetFile(trustedHostKeysFilename)
	if err != nil {
		t.Fatal(err)
	}

	for addr, hostKey := range trustedHostKeysMapping {
		t.Logf("Host Address: %d\n", addr)
		t.Logf("Host Key: %s\n\n", hostKey)
	}

}

func TestTrustedClientKeysFile(t *testing.T) {
	trustedClientKeysFilename := makeTempFile(t, "trusted_client_keys")
	trustedClientKeysFilenamePEM := trustedClientKeysFilename + ".pem"
	trustedClientKeysFilenameJWK := trustedClientKeysFilename + ".json"

	testTrustedClientKeysFile(t, trustedClientKeysFilenamePEM)
	testTrustedClientKeysFile(t, trustedClientKeysFilenameJWK)

	os.Remove(trustedClientKeysFilename)
	os.Remove(trustedClientKeysFilenamePEM)
	os.Remove(trustedClientKeysFilenameJWK)
}

func testTrustedClientKeysFile(t *testing.T, trustedClientKeysFilename string) {
	clientKey1, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	err = AddKeySetFile(trustedClientKeysFilename, clientKey1.PublicKey())
	if err != nil {
		t.Fatal(err)
	}

	trustedClientKeys, err := LoadKeySetFile(trustedClientKeysFilename)
	if err != nil {
		t.Fatal(err)
	}

	for _, clientKey := range trustedClientKeys {
		t.Logf("Client Key: %s\n", clientKey)
	}

	clientKey2, err := GenerateRSA2048PrivateKey()
	if err != nil {
		t.Fatal(err)
	}

	err = AddKeySetFile(trustedClientKeysFilename, clientKey2.PublicKey())
	if err != nil {
		t.Fatal(err)
	}

	trustedClientKeys, err = LoadKeySetFile(trustedClientKeysFilename)
	if err != nil {
		t.Fatal(err)
	}

	for _, clientKey := range trustedClientKeys {
		t.Logf("Client Key: %s\n", clientKey)
	}
}

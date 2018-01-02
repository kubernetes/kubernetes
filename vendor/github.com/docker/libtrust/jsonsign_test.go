package libtrust

import (
	"bytes"
	"crypto/rand"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"testing"

	"github.com/docker/libtrust/testutil"
)

func createTestJSON(sigKey string, indent string) (map[string]interface{}, []byte) {
	testMap := map[string]interface{}{
		"name": "dmcgowan/mycontainer",
		"config": map[string]interface{}{
			"ports": []int{9101, 9102},
			"run":   "/bin/echo \"Hello\"",
		},
		"layers": []string{
			"2893c080-27f5-11e4-8c21-0800200c9a66",
			"c54bc25b-fbb2-497b-a899-a8bc1b5b9d55",
			"4d5d7e03-f908-49f3-a7f6-9ba28dfe0fb4",
			"0b6da891-7f7f-4abf-9c97-7887549e696c",
			"1d960389-ae4f-4011-85fd-18d0f96a67ad",
		},
	}
	formattedSection := `{"config":{"ports":[9101,9102],"run":"/bin/echo \"Hello\""},"layers":["2893c080-27f5-11e4-8c21-0800200c9a66","c54bc25b-fbb2-497b-a899-a8bc1b5b9d55","4d5d7e03-f908-49f3-a7f6-9ba28dfe0fb4","0b6da891-7f7f-4abf-9c97-7887549e696c","1d960389-ae4f-4011-85fd-18d0f96a67ad"],"name":"dmcgowan/mycontainer","%s":[{"header":{`
	formattedSection = fmt.Sprintf(formattedSection, sigKey)
	if indent != "" {
		buf := bytes.NewBuffer(nil)
		json.Indent(buf, []byte(formattedSection), "", indent)
		return testMap, buf.Bytes()
	}
	return testMap, []byte(formattedSection)

}

func TestSignJSON(t *testing.T) {
	key, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("Error generating EC key: %s", err)
	}

	testMap, _ := createTestJSON("buildSignatures", "   ")
	indented, err := json.MarshalIndent(testMap, "", "   ")
	if err != nil {
		t.Fatalf("Marshall error: %s", err)
	}

	js, err := NewJSONSignature(indented)
	if err != nil {
		t.Fatalf("Error creating JSON signature: %s", err)
	}
	err = js.Sign(key)
	if err != nil {
		t.Fatalf("Error signing content: %s", err)
	}

	keys, err := js.Verify()
	if err != nil {
		t.Fatalf("Error verifying signature: %s", err)
	}
	if len(keys) != 1 {
		t.Fatalf("Error wrong number of keys returned")
	}
	if keys[0].KeyID() != key.KeyID() {
		t.Fatalf("Unexpected public key returned")
	}

}

func TestSignMap(t *testing.T) {
	key, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("Error generating EC key: %s", err)
	}

	testMap, _ := createTestJSON("buildSignatures", "   ")
	js, err := NewJSONSignatureFromMap(testMap)
	if err != nil {
		t.Fatalf("Error creating JSON signature: %s", err)
	}
	err = js.Sign(key)
	if err != nil {
		t.Fatalf("Error signing JSON signature: %s", err)
	}

	keys, err := js.Verify()
	if err != nil {
		t.Fatalf("Error verifying signature: %s", err)
	}
	if len(keys) != 1 {
		t.Fatalf("Error wrong number of keys returned")
	}
	if keys[0].KeyID() != key.KeyID() {
		t.Fatalf("Unexpected public key returned")
	}
}

func TestFormattedJson(t *testing.T) {
	key, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("Error generating EC key: %s", err)
	}

	testMap, firstSection := createTestJSON("buildSignatures", "     ")
	indented, err := json.MarshalIndent(testMap, "", "     ")
	if err != nil {
		t.Fatalf("Marshall error: %s", err)
	}

	js, err := NewJSONSignature(indented)
	if err != nil {
		t.Fatalf("Error creating JSON signature: %s", err)
	}
	err = js.Sign(key)
	if err != nil {
		t.Fatalf("Error signing content: %s", err)
	}

	b, err := js.PrettySignature("buildSignatures")
	if err != nil {
		t.Fatalf("Error signing map: %s", err)
	}

	if bytes.Compare(b[:len(firstSection)], firstSection) != 0 {
		t.Fatalf("Wrong signed value\nExpected:\n%s\nActual:\n%s", firstSection, b[:len(firstSection)])
	}

	parsed, err := ParsePrettySignature(b, "buildSignatures")
	if err != nil {
		t.Fatalf("Error parsing formatted signature: %s", err)
	}

	keys, err := parsed.Verify()
	if err != nil {
		t.Fatalf("Error verifying signature: %s", err)
	}
	if len(keys) != 1 {
		t.Fatalf("Error wrong number of keys returned")
	}
	if keys[0].KeyID() != key.KeyID() {
		t.Fatalf("Unexpected public key returned")
	}

	var unmarshalled map[string]interface{}
	err = json.Unmarshal(b, &unmarshalled)
	if err != nil {
		t.Fatalf("Could not unmarshall after parse: %s", err)
	}

}

func TestFormattedFlatJson(t *testing.T) {
	key, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("Error generating EC key: %s", err)
	}

	testMap, firstSection := createTestJSON("buildSignatures", "")
	unindented, err := json.Marshal(testMap)
	if err != nil {
		t.Fatalf("Marshall error: %s", err)
	}

	js, err := NewJSONSignature(unindented)
	if err != nil {
		t.Fatalf("Error creating JSON signature: %s", err)
	}
	err = js.Sign(key)
	if err != nil {
		t.Fatalf("Error signing JSON signature: %s", err)
	}

	b, err := js.PrettySignature("buildSignatures")
	if err != nil {
		t.Fatalf("Error signing map: %s", err)
	}

	if bytes.Compare(b[:len(firstSection)], firstSection) != 0 {
		t.Fatalf("Wrong signed value\nExpected:\n%s\nActual:\n%s", firstSection, b[:len(firstSection)])
	}

	parsed, err := ParsePrettySignature(b, "buildSignatures")
	if err != nil {
		t.Fatalf("Error parsing formatted signature: %s", err)
	}

	keys, err := parsed.Verify()
	if err != nil {
		t.Fatalf("Error verifying signature: %s", err)
	}
	if len(keys) != 1 {
		t.Fatalf("Error wrong number of keys returned")
	}
	if keys[0].KeyID() != key.KeyID() {
		t.Fatalf("Unexpected public key returned")
	}
}

func generateTrustChain(t *testing.T, key PrivateKey, ca *x509.Certificate) (PrivateKey, []*x509.Certificate) {
	parent := ca
	parentKey := key
	chain := make([]*x509.Certificate, 6)
	for i := 5; i > 0; i-- {
		intermediatekey, err := GenerateECP256PrivateKey()
		if err != nil {
			t.Fatalf("Error generate key: %s", err)
		}
		chain[i], err = testutil.GenerateIntermediate(intermediatekey.CryptoPublicKey(), parentKey.CryptoPrivateKey(), parent)
		if err != nil {
			t.Fatalf("Error generating intermdiate certificate: %s", err)
		}
		parent = chain[i]
		parentKey = intermediatekey
	}
	trustKey, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("Error generate key: %s", err)
	}
	chain[0], err = testutil.GenerateTrustCert(trustKey.CryptoPublicKey(), parentKey.CryptoPrivateKey(), parent)
	if err != nil {
		t.Fatalf("Error generate trust cert: %s", err)
	}

	return trustKey, chain
}

func TestChainVerify(t *testing.T) {
	caKey, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("Error generating key: %s", err)
	}
	ca, err := testutil.GenerateTrustCA(caKey.CryptoPublicKey(), caKey.CryptoPrivateKey())
	if err != nil {
		t.Fatalf("Error generating ca: %s", err)
	}
	trustKey, chain := generateTrustChain(t, caKey, ca)

	testMap, _ := createTestJSON("verifySignatures", "   ")
	js, err := NewJSONSignatureFromMap(testMap)
	if err != nil {
		t.Fatalf("Error creating JSONSignature from map: %s", err)
	}

	err = js.SignWithChain(trustKey, chain)
	if err != nil {
		t.Fatalf("Error signing with chain: %s", err)
	}

	pool := x509.NewCertPool()
	pool.AddCert(ca)
	chains, err := js.VerifyChains(pool)
	if err != nil {
		t.Fatalf("Error verifying content: %s", err)
	}
	if len(chains) != 1 {
		t.Fatalf("Unexpected chains length: %d", len(chains))
	}
	if len(chains[0]) != 7 {
		t.Fatalf("Unexpected chain length: %d", len(chains[0]))
	}
}

func TestInvalidChain(t *testing.T) {
	caKey, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("Error generating key: %s", err)
	}
	ca, err := testutil.GenerateTrustCA(caKey.CryptoPublicKey(), caKey.CryptoPrivateKey())
	if err != nil {
		t.Fatalf("Error generating ca: %s", err)
	}
	trustKey, chain := generateTrustChain(t, caKey, ca)

	testMap, _ := createTestJSON("verifySignatures", "   ")
	js, err := NewJSONSignatureFromMap(testMap)
	if err != nil {
		t.Fatalf("Error creating JSONSignature from map: %s", err)
	}

	err = js.SignWithChain(trustKey, chain[:5])
	if err != nil {
		t.Fatalf("Error signing with chain: %s", err)
	}

	pool := x509.NewCertPool()
	pool.AddCert(ca)
	chains, err := js.VerifyChains(pool)
	if err == nil {
		t.Fatalf("Expected error verifying with bad chain")
	}
	if len(chains) != 0 {
		t.Fatalf("Unexpected chains returned from invalid verify")
	}
}

func TestMergeSignatures(t *testing.T) {
	pk1, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("unexpected error generating private key 1: %v", err)
	}

	pk2, err := GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("unexpected error generating private key 2: %v", err)
	}

	payload := make([]byte, 1<<10)
	if _, err = io.ReadFull(rand.Reader, payload); err != nil {
		t.Fatalf("error generating payload: %v", err)
	}

	payload, _ = json.Marshal(map[string]interface{}{"data": payload})

	sig1, err := NewJSONSignature(payload)
	if err != nil {
		t.Fatalf("unexpected error creating signature 1: %v", err)
	}

	if err := sig1.Sign(pk1); err != nil {
		t.Fatalf("unexpected error signing with pk1: %v", err)
	}

	sig2, err := NewJSONSignature(payload)
	if err != nil {
		t.Fatalf("unexpected error creating signature 2: %v", err)
	}

	if err := sig2.Sign(pk2); err != nil {
		t.Fatalf("unexpected error signing with pk2: %v", err)
	}

	// Now, we actually merge into sig1
	if err := sig1.Merge(sig2); err != nil {
		t.Fatalf("unexpected error merging: %v", err)
	}

	// Verify the new signature package
	pubkeys, err := sig1.Verify()
	if err != nil {
		t.Fatalf("unexpected error during verify: %v", err)
	}

	// Make sure the pubkeys match the two private keys from before
	privkeys := map[string]PrivateKey{
		pk1.KeyID(): pk1,
		pk2.KeyID(): pk2,
	}

	found := map[string]struct{}{}

	for _, pubkey := range pubkeys {
		if _, ok := privkeys[pubkey.KeyID()]; !ok {
			t.Fatalf("unexpected public key found during verification: %v", pubkey)
		}

		found[pubkey.KeyID()] = struct{}{}
	}

	// Make sure we've found all the private keys from verification
	for keyid, _ := range privkeys {
		if _, ok := found[keyid]; !ok {
			t.Fatalf("public key %v not found during verification", keyid)
		}
	}

	// Create another signature, with a different payload, and ensure we get an error.
	sig3, err := NewJSONSignature([]byte("{}"))
	if err != nil {
		t.Fatalf("unexpected error making signature for sig3: %v", err)
	}

	if err := sig1.Merge(sig3); err == nil {
		t.Fatalf("error expected during invalid merge with different payload")
	}
}

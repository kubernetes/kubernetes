package token

import (
	"crypto"
	"crypto/rand"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/auth"
	"github.com/docker/libtrust"
)

func makeRootKeys(numKeys int) ([]libtrust.PrivateKey, error) {
	keys := make([]libtrust.PrivateKey, 0, numKeys)

	for i := 0; i < numKeys; i++ {
		key, err := libtrust.GenerateECP256PrivateKey()
		if err != nil {
			return nil, err
		}
		keys = append(keys, key)
	}

	return keys, nil
}

func makeSigningKeyWithChain(rootKey libtrust.PrivateKey, depth int) (libtrust.PrivateKey, error) {
	if depth == 0 {
		// Don't need to build a chain.
		return rootKey, nil
	}

	var (
		x5c       = make([]string, depth)
		parentKey = rootKey
		key       libtrust.PrivateKey
		cert      *x509.Certificate
		err       error
	)

	for depth > 0 {
		if key, err = libtrust.GenerateECP256PrivateKey(); err != nil {
			return nil, err
		}

		if cert, err = libtrust.GenerateCACert(parentKey, key); err != nil {
			return nil, err
		}

		depth--
		x5c[depth] = base64.StdEncoding.EncodeToString(cert.Raw)
		parentKey = key
	}

	key.AddExtendedField("x5c", x5c)

	return key, nil
}

func makeRootCerts(rootKeys []libtrust.PrivateKey) ([]*x509.Certificate, error) {
	certs := make([]*x509.Certificate, 0, len(rootKeys))

	for _, key := range rootKeys {
		cert, err := libtrust.GenerateCACert(key, key)
		if err != nil {
			return nil, err
		}
		certs = append(certs, cert)
	}

	return certs, nil
}

func makeTrustedKeyMap(rootKeys []libtrust.PrivateKey) map[string]libtrust.PublicKey {
	trustedKeys := make(map[string]libtrust.PublicKey, len(rootKeys))

	for _, key := range rootKeys {
		trustedKeys[key.KeyID()] = key.PublicKey()
	}

	return trustedKeys
}

func makeTestToken(issuer, audience string, access []*ResourceActions, rootKey libtrust.PrivateKey, depth int) (*Token, error) {
	signingKey, err := makeSigningKeyWithChain(rootKey, depth)
	if err != nil {
		return nil, fmt.Errorf("unable to make signing key with chain: %s", err)
	}

	var rawJWK json.RawMessage
	rawJWK, err = signingKey.PublicKey().MarshalJSON()
	if err != nil {
		return nil, fmt.Errorf("unable to marshal signing key to JSON: %s", err)
	}

	joseHeader := &Header{
		Type:       "JWT",
		SigningAlg: "ES256",
		RawJWK:     &rawJWK,
	}

	now := time.Now()

	randomBytes := make([]byte, 15)
	if _, err = rand.Read(randomBytes); err != nil {
		return nil, fmt.Errorf("unable to read random bytes for jwt id: %s", err)
	}

	claimSet := &ClaimSet{
		Issuer:     issuer,
		Subject:    "foo",
		Audience:   audience,
		Expiration: now.Add(5 * time.Minute).Unix(),
		NotBefore:  now.Unix(),
		IssuedAt:   now.Unix(),
		JWTID:      base64.URLEncoding.EncodeToString(randomBytes),
		Access:     access,
	}

	var joseHeaderBytes, claimSetBytes []byte

	if joseHeaderBytes, err = json.Marshal(joseHeader); err != nil {
		return nil, fmt.Errorf("unable to marshal jose header: %s", err)
	}
	if claimSetBytes, err = json.Marshal(claimSet); err != nil {
		return nil, fmt.Errorf("unable to marshal claim set: %s", err)
	}

	encodedJoseHeader := joseBase64UrlEncode(joseHeaderBytes)
	encodedClaimSet := joseBase64UrlEncode(claimSetBytes)
	encodingToSign := fmt.Sprintf("%s.%s", encodedJoseHeader, encodedClaimSet)

	var signatureBytes []byte
	if signatureBytes, _, err = signingKey.Sign(strings.NewReader(encodingToSign), crypto.SHA256); err != nil {
		return nil, fmt.Errorf("unable to sign jwt payload: %s", err)
	}

	signature := joseBase64UrlEncode(signatureBytes)
	tokenString := fmt.Sprintf("%s.%s", encodingToSign, signature)

	return NewToken(tokenString)
}

// This test makes 4 tokens with a varying number of intermediate
// certificates ranging from no intermediate chain to a length of 3
// intermediates.
func TestTokenVerify(t *testing.T) {
	var (
		numTokens = 4
		issuer    = "test-issuer"
		audience  = "test-audience"
		access    = []*ResourceActions{
			{
				Type:    "repository",
				Name:    "foo/bar",
				Actions: []string{"pull", "push"},
			},
		}
	)

	rootKeys, err := makeRootKeys(numTokens)
	if err != nil {
		t.Fatal(err)
	}

	rootCerts, err := makeRootCerts(rootKeys)
	if err != nil {
		t.Fatal(err)
	}

	rootPool := x509.NewCertPool()
	for _, rootCert := range rootCerts {
		rootPool.AddCert(rootCert)
	}

	trustedKeys := makeTrustedKeyMap(rootKeys)

	tokens := make([]*Token, 0, numTokens)

	for i := 0; i < numTokens; i++ {
		token, err := makeTestToken(issuer, audience, access, rootKeys[i], i)
		if err != nil {
			t.Fatal(err)
		}
		tokens = append(tokens, token)
	}

	verifyOps := VerifyOptions{
		TrustedIssuers:    []string{issuer},
		AcceptedAudiences: []string{audience},
		Roots:             rootPool,
		TrustedKeys:       trustedKeys,
	}

	for _, token := range tokens {
		if err := token.Verify(verifyOps); err != nil {
			t.Fatal(err)
		}
	}
}

func writeTempRootCerts(rootKeys []libtrust.PrivateKey) (filename string, err error) {
	rootCerts, err := makeRootCerts(rootKeys)
	if err != nil {
		return "", err
	}

	tempFile, err := ioutil.TempFile("", "rootCertBundle")
	if err != nil {
		return "", err
	}
	defer tempFile.Close()

	for _, cert := range rootCerts {
		if err = pem.Encode(tempFile, &pem.Block{
			Type:  "CERTIFICATE",
			Bytes: cert.Raw,
		}); err != nil {
			os.Remove(tempFile.Name())
			return "", err
		}
	}

	return tempFile.Name(), nil
}

// TestAccessController tests complete integration of the token auth package.
// It starts by mocking the options for a token auth accessController which
// it creates. It then tries a few mock requests:
// 		- don't supply a token; should error with challenge
//		- supply an invalid token; should error with challenge
// 		- supply a token with insufficient access; should error with challenge
//		- supply a valid token; should not error
func TestAccessController(t *testing.T) {
	// Make 2 keys; only the first is to be a trusted root key.
	rootKeys, err := makeRootKeys(2)
	if err != nil {
		t.Fatal(err)
	}

	rootCertBundleFilename, err := writeTempRootCerts(rootKeys[:1])
	if err != nil {
		t.Fatal(err)
	}
	defer os.Remove(rootCertBundleFilename)

	realm := "https://auth.example.com/token/"
	issuer := "test-issuer.example.com"
	service := "test-service.example.com"

	options := map[string]interface{}{
		"realm":          realm,
		"issuer":         issuer,
		"service":        service,
		"rootcertbundle": rootCertBundleFilename,
	}

	accessController, err := newAccessController(options)
	if err != nil {
		t.Fatal(err)
	}

	// 1. Make a mock http.Request with no token.
	req, err := http.NewRequest("GET", "http://example.com/foo", nil)
	if err != nil {
		t.Fatal(err)
	}

	testAccess := auth.Access{
		Resource: auth.Resource{
			Type: "foo",
			Name: "bar",
		},
		Action: "baz",
	}

	ctx := context.WithValue(nil, "http.request", req)
	authCtx, err := accessController.Authorized(ctx, testAccess)
	challenge, ok := err.(auth.Challenge)
	if !ok {
		t.Fatal("accessController did not return a challenge")
	}

	if challenge.Error() != ErrTokenRequired.Error() {
		t.Fatalf("accessControler did not get expected error - got %s - expected %s", challenge, ErrTokenRequired)
	}

	if authCtx != nil {
		t.Fatalf("expected nil auth context but got %s", authCtx)
	}

	// 2. Supply an invalid token.
	token, err := makeTestToken(
		issuer, service,
		[]*ResourceActions{{
			Type:    testAccess.Type,
			Name:    testAccess.Name,
			Actions: []string{testAccess.Action},
		}},
		rootKeys[1], 1, // Everything is valid except the key which signed it.
	)
	if err != nil {
		t.Fatal(err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token.compactRaw()))

	authCtx, err = accessController.Authorized(ctx, testAccess)
	challenge, ok = err.(auth.Challenge)
	if !ok {
		t.Fatal("accessController did not return a challenge")
	}

	if challenge.Error() != ErrInvalidToken.Error() {
		t.Fatalf("accessControler did not get expected error - got %s - expected %s", challenge, ErrTokenRequired)
	}

	if authCtx != nil {
		t.Fatalf("expected nil auth context but got %s", authCtx)
	}

	// 3. Supply a token with insufficient access.
	token, err = makeTestToken(
		issuer, service,
		[]*ResourceActions{}, // No access specified.
		rootKeys[0], 1,
	)
	if err != nil {
		t.Fatal(err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token.compactRaw()))

	authCtx, err = accessController.Authorized(ctx, testAccess)
	challenge, ok = err.(auth.Challenge)
	if !ok {
		t.Fatal("accessController did not return a challenge")
	}

	if challenge.Error() != ErrInsufficientScope.Error() {
		t.Fatalf("accessControler did not get expected error - got %s - expected %s", challenge, ErrInsufficientScope)
	}

	if authCtx != nil {
		t.Fatalf("expected nil auth context but got %s", authCtx)
	}

	// 4. Supply the token we need, or deserve, or whatever.
	token, err = makeTestToken(
		issuer, service,
		[]*ResourceActions{{
			Type:    testAccess.Type,
			Name:    testAccess.Name,
			Actions: []string{testAccess.Action},
		}},
		rootKeys[0], 1,
	)
	if err != nil {
		t.Fatal(err)
	}

	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", token.compactRaw()))

	authCtx, err = accessController.Authorized(ctx, testAccess)
	if err != nil {
		t.Fatalf("accessController returned unexpected error: %s", err)
	}

	userInfo, ok := authCtx.Value(auth.UserKey).(auth.UserInfo)
	if !ok {
		t.Fatal("token accessController did not set auth.user context")
	}

	if userInfo.Name != "foo" {
		t.Fatalf("expected user name %q, got %q", "foo", userInfo.Name)
	}
}

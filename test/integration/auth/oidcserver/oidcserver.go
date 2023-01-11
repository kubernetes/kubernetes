package oidcserver

import (
	"crypto"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/golang-jwt/jwt/v4"
	"gopkg.in/square/go-jose.v2"

	"k8s.io/apimachinery/pkg/util/uuid"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
)

type TokensType int

const (
	TestClientID     = "testclient"
	TestClientSecret = "veryrandomsecret"
	KubeAudience     = "https://kubernetes.default.svc"
)

const (
	TokensNone TokensType = iota
	TokensValid
	TokensExpiring // returns tokens that would expire in 2 seconds
	TokensExpired
	TokensInvalidSignature
	TokensError
)

type OIDCMockServer struct {
	server          *httptest.Server
	signingKey      *rsa.PrivateKey
	servingCertPath string

	mintedTokensType    TokensType
	jwks                []byte
	currentRefreshToken string
}

func RunOIDCMockServer(t *testing.T) *OIDCMockServer {
	certDir := t.TempDir()

	privateKey, err := rsa.GenerateKey(rand.Reader, 4096)
	if err != nil {
		t.Fatalf("failed to generate a signing key: %v", err)
	}

	jwksBytes, err := getJWKsBytes(privateKey)
	if err != nil {
		t.Fatalf("failed to get JWKs for private key: %v", err)
	}

	oidcServer := &OIDCMockServer{
		mintedTokensType:    TokensValid,
		jwks:                jwksBytes,
		signingKey:          privateKey,
		currentRefreshToken: "secretrefreshtoken-",
	}

	// TODO: add logging to all endpoints
	mux := http.NewServeMux()
	mux.Handle("/.well-known/openid-configuration", http.HandlerFunc(oidcServer.oidcDiscoveryHandler))
	mux.Handle("/authorization", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "AUTHZ NOT IMPLEMENTED", http.StatusTeapot)
	}))
	mux.Handle("/token", http.HandlerFunc(http.HandlerFunc(oidcServer.tokenHandler)))
	mux.Handle("/jwks", http.HandlerFunc(http.HandlerFunc(oidcServer.jwksHandler)))

	tlsServer := httptest.NewTLSServer(mux)
	// TODO: maybe post the server logs on cleanup?
	t.Cleanup(tlsServer.Close)

	oidcServer.server = tlsServer

	servingCertPEM, err := certutil.EncodeCertificates(tlsServer.Certificate())
	if err != nil {
		t.Fatalf("failed to create serving cert for the mock OIDC server: %v", err)
	}

	servingCertPath := filepath.Join(certDir, "oidc-test-serving.crt")
	if err := certutil.WriteCert(servingCertPath, servingCertPEM); err != nil {
		t.Fatalf("failed to write serving cert for the mock OIDC server: %v", err)
	}

	oidcServer.servingCertPath = servingCertPath

	return oidcServer
}

// minimalOIDCDiscovery represents the OIDC Provider Metadata defined in
// https://openid.net/specs/openid-connect-discovery-1_0.html#ProviderMetadata
//
// This struct only contains the fields that are REQUIRED per the above specification.
type minimalOIDCDiscovery struct {
	Issuer                     string   `json:"issuer"`
	AuthorizationEndpoint      string   `json:"authorization_endpoint"`
	TokenEndpoint              string   `json:"token_endpoint"` // 	This is REQUIRED unless only the Implicit Flow is used.
	JWKsURI                    string   `json:"jwks_uri"`
	ResponseTypes              []string `json:"response_types"`                        // MUST support the `code`, `id_token`, and the `token id_token` Response Type values.
	SubjectTypesSupported      []string `json:"subject_types_supported"`               // Valid types include pairwise and public.
	SupportedSigningAlgorithms []string `json:"id_token_signing_alg_values_supported"` // The algorithm RS256 MUST be included.
}

func (s *OIDCMockServer) oidcDiscoveryHandler(w http.ResponseWriter, req *http.Request) {
	resp := minimalOIDCDiscovery{
		Issuer:                     s.server.URL,
		AuthorizationEndpoint:      s.server.URL + "/authorization",
		TokenEndpoint:              s.server.URL + "/token",
		JWKsURI:                    s.server.URL + "/jwks",
		ResponseTypes:              []string{"code", "id_token", "token id_token"},
		SubjectTypesSupported:      []string{"public"},
		SupportedSigningAlgorithms: []string{"RS256"},
	}

	jsonResp, err := json.Marshal(resp)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to marshal discovery response: %v", err), http.StatusInternalServerError)
		return
	}
	w.Header().Add("Content-Type", "application/json")
	w.Write(jsonResp)
}

// minimalTokenResponse represents a successful token response as per
// https://openid.net/specs/openid-connect-core-1_0.html#TokenResponse
//
// This struct only contains the fields that are REQUIRED per the above specification
// + refresh_token
type minimalTokenResponse struct {
	AccessToken  string `json:"access_token"`
	IDToken      string `json:"id_token,omitempty"`
	TokenType    string `json:"token_type"`
	RefreshToken string `json:"refresh_token,omitempty"`
}

func (s *OIDCMockServer) SetMintedTokensType(tt TokensType) { s.mintedTokensType = tt }

// tokenHandler only implements token minting for client auth with a refresh token,
// using the client_secret_basic client auth.
// It does that rather poorly but it's all good enough for test purposes.
func (s *OIDCMockServer) tokenHandler(w http.ResponseWriter, req *http.Request) {
	if err := req.ParseForm(); err != nil || req.Method != http.MethodPost {
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write(tokenErrorResponse("invalid_request"))
		return
	}

	username, password, ok := req.BasicAuth()
	if !ok || username != TestClientID || password != TestClientSecret {
		w.Header().Add("Content-Type", "application/json")
		w.Header().Add("WWW-Authenticate", `Basic realm="oidc test"`)
		w.WriteHeader(http.StatusUnauthorized)
		w.Write(tokenErrorResponse("invalid_client"))
		return
	}

	reqForm := req.PostForm
	if reqForm.Get("grant_type") != "refresh_token" ||
		len(reqForm.Get("refresh_token")) == 0 {

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write(tokenErrorResponse("invalid_request"))
		return
	}

	if refreshToken := reqForm.Get("refresh_token"); refreshToken != s.CurrentRefreshToken() {
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		w.Write(tokenErrorResponse("invalid_grant"))
	}

	token, err := s.MintIDToken(s.mintedTokensType)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to mint an ID token: %v", err), http.StatusInternalServerError)
		return
	}

	resp := minimalTokenResponse{
		AccessToken:  "sometoken",
		IDToken:      token,
		TokenType:    "Bearer",
		RefreshToken: s.refreshedRefreshToken(),
	}

	jsonResp, err := json.Marshal(resp)
	if err != nil {
		http.Error(w, fmt.Sprintf("failed to marshal token response: %v", err), http.StatusInternalServerError)
		return
	}

	w.Header().Add("Content-Type", "application/json")
	w.Write(jsonResp)
}

func (s *OIDCMockServer) jwksHandler(w http.ResponseWriter, req *http.Request) {
	w.Header().Add("Content-Type", "application/json")
	w.Write(s.jwks)
}

type claimsWithUsername struct {
	jwt.RegisteredClaims `json:",inline"`
	Username             string `json:"username"`
}

func (c *claimsWithUsername) Valid() error {
	return c.RegisteredClaims.Valid()
}

// MintIDToken allows retrieving the token from tests without having to go through
// the client<->kube-apiserver<->oidc-server flow
// Useful for setting up the initial client conditions.
func (s *OIDCMockServer) MintIDToken(tokenType TokensType) (string, error) {
	claims := &claimsWithUsername{
		RegisteredClaims: jwt.RegisteredClaims{
			Issuer:    s.server.URL,
			Subject:   string(uuid.NewUUID()),
			Audience:  jwt.ClaimStrings{KubeAudience, TestClientID},
			ExpiresAt: jwt.NewNumericDate(time.Now().Add(1 * time.Hour)),
			IssuedAt:  jwt.NewNumericDate(time.Now().Add(-2 * time.Hour)),
		},
		Username: "testuser",
	}

	var signedToken string
	var err error
	switch tokenType {
	case TokensNone:
		signedToken = ""
	case TokensValid:
		token := jwt.NewWithClaims(jwt.GetSigningMethod("RS256"), claims)
		signedToken, err = token.SignedString(s.signingKey)
	case TokensExpired:
		claims.ExpiresAt = jwt.NewNumericDate(time.Now().Add(-1 * time.Hour))

		token := jwt.NewWithClaims(jwt.GetSigningMethod("RS256"), claims)
		signedToken, err = token.SignedString(s.signingKey)
	case TokensExpiring:
		claims.ExpiresAt = jwt.NewNumericDate(time.Now().Add(5 * time.Second))

		token := jwt.NewWithClaims(jwt.GetSigningMethod("RS256"), claims)
		signedToken, err = token.SignedString(s.signingKey)
	case TokensInvalidSignature:
		token := jwt.NewWithClaims(jwt.GetSigningMethod("RS256"), claims)

		privateKey, keyErr := rsa.GenerateKey(rand.Reader, 4096)
		if keyErr != nil {
			return "", fmt.Errorf("failed to generate a signing key: %v", keyErr)
		}
		signedToken, err = token.SignedString(privateKey)
	default:
		panic(fmt.Sprintf("%d: not implemented", tokenType))
	}

	if err != nil {
		return "", fmt.Errorf("failed to sign a token: %v", err)
	}

	return signedToken, nil
}

// OIDCConfig returns configuration for the kube-apiserver
func (s *OIDCMockServer) OIDCConfig() *kubeoptions.OIDCAuthenticationOptions {
	return &kubeoptions.OIDCAuthenticationOptions{
		IssuerURL:     s.server.URL,
		ClientID:      TestClientID,
		UsernameClaim: "username",
		CAFile:        s.servingCertPath,
	}
}

// AuthConfig returns configuration for the client
func (s *OIDCMockServer) AuthConfig() *clientcmdapi.AuthProviderConfig {
	return &clientcmdapi.AuthProviderConfig{
		Name: "oidc",
		Config: map[string]string{
			"idp-issuer-url":            s.server.URL,
			"client-id":                 TestClientID,
			"client-secret":             TestClientSecret,
			"refresh-token":             s.CurrentRefreshToken(),
			"idp-certificate-authority": s.servingCertPath,
			"id-token":                  "",
		},
	}
}

func (s *OIDCMockServer) CurrentRefreshToken() string {
	return s.currentRefreshToken
}

func (s *OIDCMockServer) refreshedRefreshToken() string {
	s.currentRefreshToken += "x"
	return s.currentRefreshToken
}

func getJWKsBytes(privKey *rsa.PrivateKey) ([]byte, error) {
	pubKey := &privKey.PublicKey
	keyId, err := keyIDFromPublicKey(pubKey)
	if err != nil {
		return nil, fmt.Errorf("failed to derive keyId from pubkey: %v", err)
	}

	jwks := jose.JSONWebKeySet{
		Keys: []jose.JSONWebKey{
			{
				Key:       pubKey,
				KeyID:     keyId,
				Algorithm: string(jose.RS256),
				Use:       "sig",
			},
		},
	}

	jwksMarshalled, err := json.Marshal(jwks)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal token response: %v", err)
	}

	return jwksMarshalled, nil
}

func keyIDFromPublicKey(publicKey interface{}) (string, error) {
	publicKeyDERBytes, err := x509.MarshalPKIXPublicKey(publicKey)
	if err != nil {
		return "", fmt.Errorf("failed to serialize public key to DER format: %v", err)
	}

	hasher := crypto.SHA256.New()
	hasher.Write(publicKeyDERBytes)
	publicKeyDERHash := hasher.Sum(nil)

	keyID := base64.RawURLEncoding.EncodeToString(publicKeyDERHash)

	return keyID, nil
}

func tokenErrorResponse(errType string) []byte {
	return []byte(fmt.Sprintf(`{"error": "%s" }`, errType))
}

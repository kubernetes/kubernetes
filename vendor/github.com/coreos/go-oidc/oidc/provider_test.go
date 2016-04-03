package oidc

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"
	"time"

	"github.com/jonboulle/clockwork"
	"github.com/kylelemons/godebug/diff"
	"github.com/kylelemons/godebug/pretty"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/oauth2"
)

func TestProviderConfigDefaults(t *testing.T) {
	var cfg ProviderConfig
	cfg = cfg.Defaults()
	tests := []struct {
		got, want []string
		name      string
	}{
		{cfg.GrantTypesSupported, DefaultGrantTypesSupported, "grant types"},
		{cfg.ResponseModesSupported, DefaultResponseModesSupported, "response modes"},
		{cfg.ClaimTypesSupported, DefaultClaimTypesSupported, "claim types"},
		{
			cfg.TokenEndpointAuthMethodsSupported,
			DefaultTokenEndpointAuthMethodsSupported,
			"token endpoint auth methods",
		},
	}

	for _, tt := range tests {
		if diff := pretty.Compare(tt.want, tt.got); diff != "" {
			t.Errorf("%s: did not match %s", tt.name, diff)
		}
	}
}

func TestProviderConfigUnmarshal(t *testing.T) {

	// helper for quickly creating uris
	uri := func(path string) *url.URL {
		return &url.URL{
			Scheme: "https",
			Host:   "server.example.com",
			Path:   path,
		}
	}

	tests := []struct {
		data    string
		want    ProviderConfig
		wantErr bool
	}{
		{
			data: `{
				"issuer": "https://server.example.com",
				"authorization_endpoint": "https://server.example.com/connect/authorize",
				"token_endpoint": "https://server.example.com/connect/token",
				"token_endpoint_auth_methods_supported": ["client_secret_basic", "private_key_jwt"],
				"token_endpoint_auth_signing_alg_values_supported": ["RS256", "ES256"],
				"userinfo_endpoint": "https://server.example.com/connect/userinfo",
				"jwks_uri": "https://server.example.com/jwks.json",
				"registration_endpoint": "https://server.example.com/connect/register",
				"scopes_supported": [
					"openid", "profile", "email", "address", "phone", "offline_access"
				],
				"response_types_supported": [
					"code", "code id_token", "id_token", "id_token token"
				],
				"acr_values_supported": [
					"urn:mace:incommon:iap:silver", "urn:mace:incommon:iap:bronze"
				],
				"subject_types_supported": ["public", "pairwise"],
				"userinfo_signing_alg_values_supported": ["RS256", "ES256", "HS256"],
				"userinfo_encryption_alg_values_supported": ["RSA1_5", "A128KW"],
				"userinfo_encryption_enc_values_supported": ["A128CBC-HS256", "A128GCM"],
				"id_token_signing_alg_values_supported": ["RS256", "ES256", "HS256"],
				"id_token_encryption_alg_values_supported": ["RSA1_5", "A128KW"],
				"id_token_encryption_enc_values_supported": ["A128CBC-HS256", "A128GCM"],
				"request_object_signing_alg_values_supported": ["none", "RS256", "ES256"],
				"display_values_supported": ["page", "popup"],
				"claim_types_supported": ["normal", "distributed"],
				"claims_supported": [
					"sub", "iss", "auth_time", "acr", "name", "given_name",
					"family_name", "nickname", "profile", "picture", "website",
					"email", "email_verified", "locale", "zoneinfo",
					"http://example.info/claims/groups"
				],
				"claims_parameter_supported": true,
				"service_documentation": "https://server.example.com/connect/service_documentation.html",
				"ui_locales_supported": ["en-US", "en-GB", "en-CA", "fr-FR", "fr-CA"]
			}
			`,
			want: ProviderConfig{
				Issuer:        &url.URL{Scheme: "https", Host: "server.example.com"},
				AuthEndpoint:  uri("/connect/authorize"),
				TokenEndpoint: uri("/connect/token"),
				TokenEndpointAuthMethodsSupported: []string{
					oauth2.AuthMethodClientSecretBasic, oauth2.AuthMethodPrivateKeyJWT,
				},
				TokenEndpointAuthSigningAlgValuesSupported: []string{
					jose.AlgRS256, jose.AlgES256,
				},
				UserInfoEndpoint:     uri("/connect/userinfo"),
				KeysEndpoint:         uri("/jwks.json"),
				RegistrationEndpoint: uri("/connect/register"),
				ScopesSupported: []string{
					"openid", "profile", "email", "address", "phone", "offline_access",
				},
				ResponseTypesSupported: []string{
					oauth2.ResponseTypeCode, oauth2.ResponseTypeCodeIDToken,
					oauth2.ResponseTypeIDToken, oauth2.ResponseTypeIDTokenToken,
				},
				ACRValuesSupported: []string{
					"urn:mace:incommon:iap:silver", "urn:mace:incommon:iap:bronze",
				},
				SubjectTypesSupported: []string{
					SubjectTypePublic, SubjectTypePairwise,
				},
				UserInfoSigningAlgValues:    []string{jose.AlgRS256, jose.AlgES256, jose.AlgHS256},
				UserInfoEncryptionAlgValues: []string{"RSA1_5", "A128KW"},
				UserInfoEncryptionEncValues: []string{"A128CBC-HS256", "A128GCM"},
				IDTokenSigningAlgValues:     []string{jose.AlgRS256, jose.AlgES256, jose.AlgHS256},
				IDTokenEncryptionAlgValues:  []string{"RSA1_5", "A128KW"},
				IDTokenEncryptionEncValues:  []string{"A128CBC-HS256", "A128GCM"},
				ReqObjSigningAlgValues:      []string{jose.AlgNone, jose.AlgRS256, jose.AlgES256},
				DisplayValuesSupported:      []string{"page", "popup"},
				ClaimTypesSupported:         []string{"normal", "distributed"},
				ClaimsSupported: []string{
					"sub", "iss", "auth_time", "acr", "name", "given_name",
					"family_name", "nickname", "profile", "picture", "website",
					"email", "email_verified", "locale", "zoneinfo",
					"http://example.info/claims/groups",
				},
				ClaimsParameterSupported: true,
				ServiceDocs:              uri("/connect/service_documentation.html"),
				UILocalsSupported:        []string{"en-US", "en-GB", "en-CA", "fr-FR", "fr-CA"},
			},
			wantErr: false,
		},
		{
			// missing a lot of required field
			data:    `{}`,
			wantErr: true,
		},
		{
			data: `{
				"issuer": "https://server.example.com",
				"authorization_endpoint": "https://server.example.com/connect/authorize",
				"token_endpoint": "https://server.example.com/connect/token",
				"jwks_uri": "https://server.example.com/jwks.json",
				"response_types_supported": [
					"code", "code id_token", "id_token", "id_token token"
				],
				"subject_types_supported": ["public", "pairwise"],
				"id_token_signing_alg_values_supported": ["RS256", "ES256", "HS256"]
			}
			`,
			want: ProviderConfig{
				Issuer:        &url.URL{Scheme: "https", Host: "server.example.com"},
				AuthEndpoint:  uri("/connect/authorize"),
				TokenEndpoint: uri("/connect/token"),
				KeysEndpoint:  uri("/jwks.json"),
				ResponseTypesSupported: []string{
					oauth2.ResponseTypeCode, oauth2.ResponseTypeCodeIDToken,
					oauth2.ResponseTypeIDToken, oauth2.ResponseTypeIDTokenToken,
				},
				SubjectTypesSupported: []string{
					SubjectTypePublic, SubjectTypePairwise,
				},
				IDTokenSigningAlgValues: []string{jose.AlgRS256, jose.AlgES256, jose.AlgHS256},
			},
			wantErr: false,
		},
		{
			// invalid scheme 'ftp://'
			data: `{
				"issuer": "https://server.example.com",
				"authorization_endpoint": "https://server.example.com/connect/authorize",
				"token_endpoint": "https://server.example.com/connect/token",
				"jwks_uri": "ftp://server.example.com/jwks.json",
				"response_types_supported": [
					"code", "code id_token", "id_token", "id_token token"
				],
				"subject_types_supported": ["public", "pairwise"],
				"id_token_signing_alg_values_supported": ["RS256", "ES256", "HS256"]
			}
			`,
			wantErr: true,
		},
	}
	for i, tt := range tests {
		var got ProviderConfig
		if err := json.Unmarshal([]byte(tt.data), &got); err != nil {
			if !tt.wantErr {
				t.Errorf("case %d: failed to unmarshal provider config: %v", i, err)
			}
			continue
		}
		if tt.wantErr {
			t.Errorf("case %d: expected error", i)
			continue
		}
		if diff := pretty.Compare(tt.want, got); diff != "" {
			t.Errorf("case %d: unmarshaled struct did not match expected %s", i, diff)
		}
	}

}

func TestProviderConfigMarshal(t *testing.T) {
	tests := []struct {
		cfg  ProviderConfig
		want string
	}{
		{
			cfg: ProviderConfig{
				Issuer: &url.URL{Scheme: "https", Host: "auth.example.com"},
				AuthEndpoint: &url.URL{
					Scheme: "https", Host: "auth.example.com", Path: "/auth",
				},
				TokenEndpoint: &url.URL{
					Scheme: "https", Host: "auth.example.com", Path: "/token",
				},
				UserInfoEndpoint: &url.URL{
					Scheme: "https", Host: "auth.example.com", Path: "/userinfo",
				},
				KeysEndpoint: &url.URL{
					Scheme: "https", Host: "auth.example.com", Path: "/jwk",
				},
				ResponseTypesSupported:  []string{oauth2.ResponseTypeCode},
				SubjectTypesSupported:   []string{SubjectTypePublic},
				IDTokenSigningAlgValues: []string{jose.AlgRS256},
			},
			// spacing must match json.MarshalIndent(cfg, "", "\t")
			want: `{
	"issuer": "https://auth.example.com",
	"authorization_endpoint": "https://auth.example.com/auth",
	"token_endpoint": "https://auth.example.com/token",
	"userinfo_endpoint": "https://auth.example.com/userinfo",
	"jwks_uri": "https://auth.example.com/jwk",
	"response_types_supported": [
		"code"
	],
	"subject_types_supported": [
		"public"
	],
	"id_token_signing_alg_values_supported": [
		"RS256"
	]
}`,
		},
		{
			cfg: ProviderConfig{
				Issuer: &url.URL{Scheme: "https", Host: "auth.example.com"},
				AuthEndpoint: &url.URL{
					Scheme: "https", Host: "auth.example.com", Path: "/auth",
				},
				TokenEndpoint: &url.URL{
					Scheme: "https", Host: "auth.example.com", Path: "/token",
				},
				UserInfoEndpoint: &url.URL{
					Scheme: "https", Host: "auth.example.com", Path: "/userinfo",
				},
				KeysEndpoint: &url.URL{
					Scheme: "https", Host: "auth.example.com", Path: "/jwk",
				},
				RegistrationEndpoint: &url.URL{
					Scheme: "https", Host: "auth.example.com", Path: "/register",
				},
				ScopesSupported:         DefaultScope,
				ResponseTypesSupported:  []string{oauth2.ResponseTypeCode},
				ResponseModesSupported:  DefaultResponseModesSupported,
				GrantTypesSupported:     []string{oauth2.GrantTypeAuthCode},
				SubjectTypesSupported:   []string{SubjectTypePublic},
				IDTokenSigningAlgValues: []string{jose.AlgRS256},
				ServiceDocs:             &url.URL{Scheme: "https", Host: "example.com", Path: "/docs"},
			},
			// spacing must match json.MarshalIndent(cfg, "", "\t")
			want: `{
	"issuer": "https://auth.example.com",
	"authorization_endpoint": "https://auth.example.com/auth",
	"token_endpoint": "https://auth.example.com/token",
	"userinfo_endpoint": "https://auth.example.com/userinfo",
	"jwks_uri": "https://auth.example.com/jwk",
	"registration_endpoint": "https://auth.example.com/register",
	"scopes_supported": [
		"openid",
		"email",
		"profile"
	],
	"response_types_supported": [
		"code"
	],
	"response_modes_supported": [
		"query",
		"fragment"
	],
	"grant_types_supported": [
		"authorization_code"
	],
	"subject_types_supported": [
		"public"
	],
	"id_token_signing_alg_values_supported": [
		"RS256"
	],
	"service_documentation": "https://example.com/docs"
}`,
		},
	}

	for i, tt := range tests {
		got, err := json.MarshalIndent(&tt.cfg, "", "\t")
		if err != nil {
			t.Errorf("case %d: failed to marshal config: %v", i, err)
			continue
		}
		if d := diff.Diff(string(got), string(tt.want)); d != "" {
			t.Errorf("case %d: expected did not match result: %s", i, d)
		}

		var cfg ProviderConfig
		if err := json.Unmarshal(got, &cfg); err != nil {
			t.Errorf("case %d: could not unmarshal marshal response: %v", i, err)
			continue
		}

		if d := pretty.Compare(tt.cfg, cfg); d != "" {
			t.Errorf("case %d: config did not survive JSON marshaling round trip: %s", i, d)
		}
	}

}

func TestProviderConfigSupports(t *testing.T) {
	tests := []struct {
		provider                   ProviderConfig
		client                     ClientMetadata
		fillRequiredProviderFields bool
		ok                         bool
	}{
		{
			provider: ProviderConfig{},
			client: ClientMetadata{
				RedirectURIs: []url.URL{
					{Scheme: "https", Host: "example.com", Path: "/callback"},
				},
			},
			fillRequiredProviderFields: true,
			ok: true,
		},
		{
			// invalid provider config
			provider: ProviderConfig{},
			client: ClientMetadata{
				RedirectURIs: []url.URL{
					{Scheme: "https", Host: "example.com", Path: "/callback"},
				},
			},
			fillRequiredProviderFields: false,
			ok: false,
		},
		{
			// invalid client config
			provider: ProviderConfig{},
			client:   ClientMetadata{},
			fillRequiredProviderFields: true,
			ok: false,
		},
	}

	for i, tt := range tests {
		if tt.fillRequiredProviderFields {
			tt.provider = fillRequiredProviderFields(tt.provider)
		}

		err := tt.provider.Supports(tt.client)
		if err == nil && !tt.ok {
			t.Errorf("case %d: expected non-nil error", i)
		}
		if err != nil && tt.ok {
			t.Errorf("case %d: supports failed: %v", i, err)
		}
	}
}

func newValidProviderConfig() ProviderConfig {
	var cfg ProviderConfig
	return fillRequiredProviderFields(cfg)
}

// fill a provider config with enough information to be valid
func fillRequiredProviderFields(cfg ProviderConfig) ProviderConfig {
	if cfg.Issuer == nil {
		cfg.Issuer = &url.URL{Scheme: "https", Host: "auth.example.com"}
	}
	urlPath := func(path string) *url.URL {
		var u url.URL
		u = *cfg.Issuer
		u.Path = path
		return &u
	}
	cfg.AuthEndpoint = urlPath("/auth")
	cfg.TokenEndpoint = urlPath("/token")
	cfg.UserInfoEndpoint = urlPath("/userinfo")
	cfg.KeysEndpoint = urlPath("/jwk")
	cfg.ResponseTypesSupported = []string{oauth2.ResponseTypeCode}
	cfg.SubjectTypesSupported = []string{SubjectTypePublic}
	cfg.IDTokenSigningAlgValues = []string{jose.AlgRS256}
	return cfg
}

type fakeProviderConfigGetterSetter struct {
	cfg      *ProviderConfig
	getCount int
	setCount int
}

func (g *fakeProviderConfigGetterSetter) Get() (ProviderConfig, error) {
	g.getCount++
	return *g.cfg, nil
}

func (g *fakeProviderConfigGetterSetter) Set(cfg ProviderConfig) error {
	g.cfg = &cfg
	g.setCount++
	return nil
}

type fakeProviderConfigHandler struct {
	cfg    ProviderConfig
	maxAge time.Duration
}

func (s *fakeProviderConfigHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := json.Marshal(&s.cfg)
	if s.maxAge.Seconds() >= 0 {
		w.Header().Set("Cache-Control", fmt.Sprintf("public, max-age=%d", int(s.maxAge.Seconds())))
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(b)
}

func TestProviderConfigRequiredFields(t *testing.T) {
	// Ensure provider metadata responses have all the required fields.
	// taken from https://openid.net/specs/openid-connect-discovery-1_0.html#ProviderMetadata
	requiredFields := []string{
		"issuer",
		"authorization_endpoint",
		"token_endpoint", // "This is REQUIRED unless only the Implicit Flow is used."
		"jwks_uri",
		"response_types_supported",
		"subject_types_supported",
		"id_token_signing_alg_values_supported",
	}

	svr := &fakeProviderConfigHandler{
		cfg: ProviderConfig{
			Issuer:    &url.URL{Scheme: "http", Host: "example.com"},
			ExpiresAt: time.Now().Add(time.Minute),
		},
		maxAge: time.Minute,
	}
	svr.cfg = fillRequiredProviderFields(svr.cfg)
	s := httptest.NewServer(svr)
	defer s.Close()

	resp, err := http.Get(s.URL + "/")
	if err != nil {
		t.Errorf("get: %v", err)
		return
	}
	defer resp.Body.Close()
	var data map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		t.Errorf("decode: %v", err)
		return
	}
	for _, field := range requiredFields {
		if _, ok := data[field]; !ok {
			t.Errorf("provider metadata does not have required field '%s'", field)
		}
	}
}

type handlerClient struct {
	Handler http.Handler
}

func (hc *handlerClient) Do(r *http.Request) (*http.Response, error) {
	w := httptest.NewRecorder()
	hc.Handler.ServeHTTP(w, r)

	resp := http.Response{
		StatusCode: w.Code,
		Header:     w.Header(),
		Body:       ioutil.NopCloser(w.Body),
	}

	return &resp, nil
}

func TestHTTPProviderConfigGetter(t *testing.T) {
	svr := &fakeProviderConfigHandler{}
	hc := &handlerClient{Handler: svr}
	fc := clockwork.NewFakeClock()
	now := fc.Now().UTC()

	tests := []struct {
		dsc string
		age time.Duration
		cfg ProviderConfig
		ok  bool
	}{
		// everything is good
		{
			dsc: "https://example.com",
			age: time.Minute,
			cfg: ProviderConfig{
				Issuer:    &url.URL{Scheme: "https", Host: "example.com"},
				ExpiresAt: now.Add(time.Minute),
			},
			ok: true,
		},
		// iss and disco url differ by scheme only (how google works)
		{
			dsc: "https://example.com",
			age: time.Minute,
			cfg: ProviderConfig{
				Issuer:    &url.URL{Scheme: "https", Host: "example.com"},
				ExpiresAt: now.Add(time.Minute),
			},
			ok: true,
		},
		// issuer and discovery URL mismatch
		{
			dsc: "https://foo.com",
			age: time.Minute,
			cfg: ProviderConfig{
				Issuer:    &url.URL{Scheme: "https", Host: "example.com"},
				ExpiresAt: now.Add(time.Minute),
			},
			ok: false,
		},
		// missing cache header results in zero ExpiresAt
		{
			dsc: "https://example.com",
			age: -1,
			cfg: ProviderConfig{
				Issuer: &url.URL{Scheme: "https", Host: "example.com"},
			},
			ok: true,
		},
	}

	for i, tt := range tests {
		tt.cfg = fillRequiredProviderFields(tt.cfg)
		svr.cfg = tt.cfg
		svr.maxAge = tt.age
		getter := NewHTTPProviderConfigGetter(hc, tt.dsc)
		getter.clock = fc

		got, err := getter.Get()
		if err != nil {
			if tt.ok {
				t.Errorf("test %d: unexpected error: %v", i, err)
			}
			continue
		}

		if !tt.ok {
			t.Errorf("test %d: expected error", i)
			continue
		}

		if !reflect.DeepEqual(tt.cfg, got) {
			t.Errorf("test %d: want: %#v, got: %#v", i, tt.cfg, got)
		}
	}
}

func TestProviderConfigSyncerRun(t *testing.T) {
	c1 := &ProviderConfig{
		Issuer: &url.URL{Scheme: "https", Host: "example.com"},
	}
	c2 := &ProviderConfig{
		Issuer: &url.URL{Scheme: "https", Host: "example.com"},
	}

	tests := []struct {
		first     *ProviderConfig
		advance   time.Duration
		second    *ProviderConfig
		firstExp  time.Duration
		secondExp time.Duration
		count     int
	}{
		// exp is 10m, should have same config after 1s
		{
			first:     c1,
			firstExp:  time.Duration(10 * time.Minute),
			advance:   time.Minute,
			second:    c1,
			secondExp: time.Duration(10 * time.Minute),
			count:     1,
		},
		// exp is 10m, should have new config after 10/2 = 5m
		{
			first:     c1,
			firstExp:  time.Duration(10 * time.Minute),
			advance:   time.Duration(5 * time.Minute),
			second:    c2,
			secondExp: time.Duration(10 * time.Minute),
			count:     2,
		},
		// exp is 20m, should have new config after 20/2 = 10m
		{
			first:     c1,
			firstExp:  time.Duration(20 * time.Minute),
			advance:   time.Duration(10 * time.Minute),
			second:    c2,
			secondExp: time.Duration(30 * time.Minute),
			count:     2,
		},
	}

	assertCfg := func(i int, to *fakeProviderConfigGetterSetter, want ProviderConfig) {
		got, err := to.Get()
		if err != nil {
			t.Fatalf("test %d: unable to get config: %v", i, err)
		}
		if !reflect.DeepEqual(want, got) {
			t.Fatalf("test %d: incorrect state:\nwant=%#v\ngot=%#v", i, want, got)
		}
	}

	for i, tt := range tests {
		from := &fakeProviderConfigGetterSetter{}
		to := &fakeProviderConfigGetterSetter{}

		fc := clockwork.NewFakeClock()
		now := fc.Now().UTC()
		syncer := NewProviderConfigSyncer(from, to)
		syncer.clock = fc

		tt.first.ExpiresAt = now.Add(tt.firstExp)
		tt.second.ExpiresAt = now.Add(tt.secondExp)
		if err := from.Set(*tt.first); err != nil {
			t.Fatalf("test %d: unexpected error: %v", i, err)
		}

		stop := syncer.Run()
		defer close(stop)
		fc.BlockUntil(1)

		// first sync
		assertCfg(i, to, *tt.first)

		if err := from.Set(*tt.second); err != nil {
			t.Fatalf("test %d: unexpected error: %v", i, err)
		}

		fc.Advance(tt.advance)
		fc.BlockUntil(1)

		// second sync
		assertCfg(i, to, *tt.second)

		if tt.count != from.getCount {
			t.Fatalf("test %d: want: %v, got: %v", i, tt.count, from.getCount)
		}
	}
}

type staticProviderConfigGetter struct {
	cfg ProviderConfig
	err error
}

func (g *staticProviderConfigGetter) Get() (ProviderConfig, error) {
	return g.cfg, g.err
}

type staticProviderConfigSetter struct {
	cfg *ProviderConfig
	err error
}

func (s *staticProviderConfigSetter) Set(cfg ProviderConfig) error {
	s.cfg = &cfg
	return s.err
}

func TestProviderConfigSyncerSyncFailure(t *testing.T) {
	fc := clockwork.NewFakeClock()

	tests := []struct {
		from *staticProviderConfigGetter
		to   *staticProviderConfigSetter

		// want indicates what ProviderConfig should be passed to Set.
		// If nil, the Set should not be called.
		want *ProviderConfig
	}{
		// generic Get failure
		{
			from: &staticProviderConfigGetter{err: errors.New("fail")},
			to:   &staticProviderConfigSetter{},
			want: nil,
		},
		// generic Set failure
		{
			from: &staticProviderConfigGetter{cfg: ProviderConfig{ExpiresAt: fc.Now().Add(time.Minute)}},
			to:   &staticProviderConfigSetter{err: errors.New("fail")},
			want: &ProviderConfig{ExpiresAt: fc.Now().Add(time.Minute)},
		},
	}

	for i, tt := range tests {
		pcs := &ProviderConfigSyncer{
			from:  tt.from,
			to:    tt.to,
			clock: fc,
		}
		_, err := pcs.sync()
		if err == nil {
			t.Errorf("case %d: expected non-nil error", i)
		}
		if !reflect.DeepEqual(tt.want, tt.to.cfg) {
			t.Errorf("case %d: Set mismatch: want=%#v got=%#v", i, tt.want, tt.to.cfg)
		}
	}
}

func TestNextSyncAfter(t *testing.T) {
	fc := clockwork.NewFakeClock()

	tests := []struct {
		exp  time.Time
		want time.Duration
	}{
		{
			exp:  fc.Now().Add(time.Hour),
			want: 30 * time.Minute,
		},
		// override large values with the maximum
		{
			exp:  fc.Now().Add(168 * time.Hour), // one week
			want: 24 * time.Hour,
		},
		// override "now" values with the minimum
		{
			exp:  fc.Now(),
			want: time.Minute,
		},
		// override negative values with the minimum
		{
			exp:  fc.Now().Add(-1 * time.Minute),
			want: time.Minute,
		},
		// zero-value Time results in maximum sync interval
		{
			exp:  time.Time{},
			want: 24 * time.Hour,
		},
	}

	for i, tt := range tests {
		got := nextSyncAfter(tt.exp, fc)
		if tt.want != got {
			t.Errorf("case %d: want=%v got=%v", i, tt.want, got)
		}
	}
}

func TestProviderConfigEmpty(t *testing.T) {
	cfg := ProviderConfig{}
	if !cfg.Empty() {
		t.Fatalf("Empty provider config reports non-empty")
	}
	cfg = ProviderConfig{
		Issuer: &url.URL{Scheme: "https", Host: "example.com"},
	}
	if cfg.Empty() {
		t.Fatalf("Non-empty provider config reports empty")
	}
}

func TestPCSStepAfter(t *testing.T) {
	pass := func() (time.Duration, error) { return 7 * time.Second, nil }
	fail := func() (time.Duration, error) { return 0, errors.New("fail") }

	tests := []struct {
		stepper  pcsStepper
		stepFunc pcsStepFunc
		want     pcsStepper
	}{
		// good step results in retry at TTL
		{
			stepper:  &pcsStepNext{},
			stepFunc: pass,
			want:     &pcsStepNext{aft: 7 * time.Second},
		},

		// good step after failed step results results in retry at TTL
		{
			stepper:  &pcsStepRetry{aft: 2 * time.Second},
			stepFunc: pass,
			want:     &pcsStepNext{aft: 7 * time.Second},
		},

		// failed step results in a retry in 1s
		{
			stepper:  &pcsStepNext{},
			stepFunc: fail,
			want:     &pcsStepRetry{aft: time.Second},
		},

		// failed retry backs off by a factor of 2
		{
			stepper:  &pcsStepRetry{aft: time.Second},
			stepFunc: fail,
			want:     &pcsStepRetry{aft: 2 * time.Second},
		},

		// failed retry backs off by a factor of 2, up to 1m
		{
			stepper:  &pcsStepRetry{aft: 32 * time.Second},
			stepFunc: fail,
			want:     &pcsStepRetry{aft: 60 * time.Second},
		},
	}

	for i, tt := range tests {
		got := tt.stepper.step(tt.stepFunc)
		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf("case %d: want=%#v got=%#v", i, tt.want, got)
		}
	}
}

func TestProviderConfigSupportsGrantType(t *testing.T) {
	tests := []struct {
		types []string
		typ   string
		want  bool
	}{
		// explicitly supported
		{
			types: []string{"foo_type"},
			typ:   "foo_type",
			want:  true,
		},

		// explicitly unsupported
		{
			types: []string{"bar_type"},
			typ:   "foo_type",
			want:  false,
		},

		// default type explicitly unsupported
		{
			types: []string{oauth2.GrantTypeImplicit},
			typ:   oauth2.GrantTypeAuthCode,
			want:  false,
		},

		// type not found in default set
		{
			types: []string{},
			typ:   "foo_type",
			want:  false,
		},

		// type found in default set
		{
			types: []string{},
			typ:   oauth2.GrantTypeAuthCode,
			want:  true,
		},
	}

	for i, tt := range tests {
		cfg := ProviderConfig{
			GrantTypesSupported: tt.types,
		}
		got := cfg.SupportsGrantType(tt.typ)
		if tt.want != got {
			t.Errorf("case %d: assert %v supports %v: want=%t got=%t", i, tt.types, tt.typ, tt.want, got)
		}
	}
}

type fakeClient struct {
	resp *http.Response
}

func (f *fakeClient) Do(req *http.Request) (*http.Response, error) {
	return f.resp, nil
}

func TestWaitForProviderConfigImmediateSuccess(t *testing.T) {
	cfg := newValidProviderConfig()
	b, err := json.Marshal(&cfg)
	if err != nil {
		t.Fatalf("Failed marshaling provider config")
	}

	resp := http.Response{Body: ioutil.NopCloser(bytes.NewBuffer(b))}
	hc := &fakeClient{&resp}
	fc := clockwork.NewFakeClock()

	reschan := make(chan ProviderConfig)
	go func() {
		reschan <- waitForProviderConfig(hc, cfg.Issuer.String(), fc)
	}()

	var got ProviderConfig
	select {
	case got = <-reschan:
	case <-time.After(time.Second):
		t.Fatalf("Did not receive result within 1s")
	}

	if !reflect.DeepEqual(cfg, got) {
		t.Fatalf("Received incorrect provider config: want=%#v got=%#v", cfg, got)
	}
}

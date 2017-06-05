package oidc

import (
	"encoding/json"
	"net/mail"
	"net/url"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/key"
	"github.com/coreos/go-oidc/oauth2"
	"github.com/kylelemons/godebug/pretty"
)

func TestNewClientScopeDefault(t *testing.T) {
	tests := []struct {
		c ClientConfig
		e []string
	}{
		{
			// No scope
			c: ClientConfig{RedirectURL: "http://example.com/redirect"},
			e: DefaultScope,
		},
		{
			// Nil scope
			c: ClientConfig{RedirectURL: "http://example.com/redirect", Scope: nil},
			e: DefaultScope,
		},
		{
			// Empty scope
			c: ClientConfig{RedirectURL: "http://example.com/redirect", Scope: []string{}},
			e: []string{},
		},
		{
			// Custom scope equal to default
			c: ClientConfig{RedirectURL: "http://example.com/redirect", Scope: []string{"openid", "email", "profile"}},
			e: DefaultScope,
		},
		{
			// Custom scope not including defaults
			c: ClientConfig{RedirectURL: "http://example.com/redirect", Scope: []string{"foo", "bar"}},
			e: []string{"foo", "bar"},
		},
		{
			// Custom scopes overlapping with defaults
			c: ClientConfig{RedirectURL: "http://example.com/redirect", Scope: []string{"openid", "foo"}},
			e: []string{"openid", "foo"},
		},
	}

	for i, tt := range tests {
		c, err := NewClient(tt.c)
		if err != nil {
			t.Errorf("case %d: unexpected error from NewClient: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(tt.e, c.scope) {
			t.Errorf("case %d: want: %v, got: %v", i, tt.e, c.scope)
		}
	}
}

func TestHealthy(t *testing.T) {
	now := time.Now().UTC()

	tests := []struct {
		p ProviderConfig
		h bool
	}{
		// all ok
		{
			p: ProviderConfig{
				Issuer:    &url.URL{Scheme: "http", Host: "example.com"},
				ExpiresAt: now.Add(time.Hour),
			},
			h: true,
		},
		// zero-value ProviderConfig.ExpiresAt
		{
			p: ProviderConfig{
				Issuer: &url.URL{Scheme: "http", Host: "example.com"},
			},
			h: true,
		},
		// expired ProviderConfig
		{
			p: ProviderConfig{
				Issuer:    &url.URL{Scheme: "http", Host: "example.com"},
				ExpiresAt: now.Add(time.Hour * -1),
			},
			h: false,
		},
		// empty ProviderConfig
		{
			p: ProviderConfig{},
			h: false,
		},
	}

	for i, tt := range tests {
		c := &Client{providerConfig: newProviderConfigRepo(tt.p)}
		err := c.Healthy()
		want := tt.h
		got := (err == nil)

		if want != got {
			t.Errorf("case %d: want: healthy=%v, got: healhty=%v, err: %v", i, want, got, err)
		}
	}
}

func TestClientKeysFuncAll(t *testing.T) {
	priv1, err := key.GeneratePrivateKey()
	if err != nil {
		t.Fatalf("failed to generate private key, error=%v", err)
	}

	priv2, err := key.GeneratePrivateKey()
	if err != nil {
		t.Fatalf("failed to generate private key, error=%v", err)
	}

	now := time.Now()
	future := now.Add(time.Hour)
	past := now.Add(-1 * time.Hour)

	tests := []struct {
		keySet *key.PublicKeySet
		want   []key.PublicKey
	}{
		// two keys, non-expired set
		{
			keySet: key.NewPublicKeySet([]jose.JWK{priv2.JWK(), priv1.JWK()}, future),
			want:   []key.PublicKey{*key.NewPublicKey(priv2.JWK()), *key.NewPublicKey(priv1.JWK())},
		},

		// no keys, non-expired set
		{
			keySet: key.NewPublicKeySet([]jose.JWK{}, future),
			want:   []key.PublicKey{},
		},

		// two keys, expired set
		{
			keySet: key.NewPublicKeySet([]jose.JWK{priv2.JWK(), priv1.JWK()}, past),
			want:   []key.PublicKey{},
		},

		// no keys, expired set
		{
			keySet: key.NewPublicKeySet([]jose.JWK{}, past),
			want:   []key.PublicKey{},
		},
	}

	for i, tt := range tests {
		var c Client
		c.keySet = *tt.keySet
		keysFunc := c.keysFuncAll()
		got := keysFunc()
		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf("case %d: want=%#v got=%#v", i, tt.want, got)
		}
	}
}

func TestClientKeysFuncWithID(t *testing.T) {
	priv1, err := key.GeneratePrivateKey()
	if err != nil {
		t.Fatalf("failed to generate private key, error=%v", err)
	}

	priv2, err := key.GeneratePrivateKey()
	if err != nil {
		t.Fatalf("failed to generate private key, error=%v", err)
	}

	now := time.Now()
	future := now.Add(time.Hour)
	past := now.Add(-1 * time.Hour)

	tests := []struct {
		keySet *key.PublicKeySet
		argID  string
		want   []key.PublicKey
	}{
		// two keys, match, non-expired set
		{
			keySet: key.NewPublicKeySet([]jose.JWK{priv2.JWK(), priv1.JWK()}, future),
			argID:  priv2.ID(),
			want:   []key.PublicKey{*key.NewPublicKey(priv2.JWK())},
		},

		// two keys, no match, non-expired set
		{
			keySet: key.NewPublicKeySet([]jose.JWK{priv2.JWK(), priv1.JWK()}, future),
			argID:  "XXX",
			want:   []key.PublicKey{},
		},

		// no keys, no match, non-expired set
		{
			keySet: key.NewPublicKeySet([]jose.JWK{}, future),
			argID:  priv2.ID(),
			want:   []key.PublicKey{},
		},

		// two keys, match, expired set
		{
			keySet: key.NewPublicKeySet([]jose.JWK{priv2.JWK(), priv1.JWK()}, past),
			argID:  priv2.ID(),
			want:   []key.PublicKey{},
		},

		// no keys, no match, expired set
		{
			keySet: key.NewPublicKeySet([]jose.JWK{}, past),
			argID:  priv2.ID(),
			want:   []key.PublicKey{},
		},
	}

	for i, tt := range tests {
		var c Client
		c.keySet = *tt.keySet
		keysFunc := c.keysFuncWithID(tt.argID)
		got := keysFunc()
		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf("case %d: want=%#v got=%#v", i, tt.want, got)
		}
	}
}

func TestClientMetadataValid(t *testing.T) {
	tests := []ClientMetadata{
		// one RedirectURL
		ClientMetadata{
			RedirectURIs: []url.URL{url.URL{Scheme: "http", Host: "example.com"}},
		},

		// one RedirectURL w/ nonempty path
		ClientMetadata{
			RedirectURIs: []url.URL{url.URL{Scheme: "http", Host: "example.com", Path: "/foo"}},
		},

		// two RedirectURIs
		ClientMetadata{
			RedirectURIs: []url.URL{
				url.URL{Scheme: "http", Host: "foo.example.com"},
				url.URL{Scheme: "http", Host: "bar.example.com"},
			},
		},
	}

	for i, tt := range tests {
		if err := tt.Valid(); err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
		}
	}
}

func TestClientMetadataInvalid(t *testing.T) {
	tests := []ClientMetadata{
		// nil RedirectURls slice
		ClientMetadata{
			RedirectURIs: nil,
		},

		// empty RedirectURIs slice
		ClientMetadata{
			RedirectURIs: []url.URL{},
		},

		// empty url.URL
		ClientMetadata{
			RedirectURIs: []url.URL{url.URL{}},
		},

		// empty url.URL following OK item
		ClientMetadata{
			RedirectURIs: []url.URL{url.URL{Scheme: "http", Host: "example.com"}, url.URL{}},
		},

		// url.URL with empty Host
		ClientMetadata{
			RedirectURIs: []url.URL{url.URL{Scheme: "http", Host: ""}},
		},

		// url.URL with empty Scheme
		ClientMetadata{
			RedirectURIs: []url.URL{url.URL{Scheme: "", Host: "example.com"}},
		},

		// url.URL with non-HTTP(S) Scheme
		ClientMetadata{
			RedirectURIs: []url.URL{url.URL{Scheme: "tcp", Host: "127.0.0.1"}},
		},

		// EncryptionEnc without EncryptionAlg
		ClientMetadata{
			RedirectURIs: []url.URL{url.URL{Scheme: "http", Host: "example.com"}},
			IDTokenResponseOptions: JWAOptions{
				EncryptionEnc: "A128CBC-HS256",
			},
		},

		// List of URIs with one empty element
		ClientMetadata{
			RedirectURIs: []url.URL{url.URL{Scheme: "http", Host: "example.com"}},
			RequestURIs: []url.URL{
				url.URL{Scheme: "http", Host: "example.com"},
				url.URL{},
			},
		},
	}

	for i, tt := range tests {
		if err := tt.Valid(); err == nil {
			t.Errorf("case %d: expected non-nil error", i)
		}
	}
}

func TestChooseAuthMethod(t *testing.T) {
	tests := []struct {
		supported []string
		chosen    string
		err       bool
	}{
		{
			supported: []string{},
			chosen:    oauth2.AuthMethodClientSecretBasic,
		},
		{
			supported: []string{oauth2.AuthMethodClientSecretBasic},
			chosen:    oauth2.AuthMethodClientSecretBasic,
		},
		{
			supported: []string{oauth2.AuthMethodClientSecretPost},
			chosen:    oauth2.AuthMethodClientSecretPost,
		},
		{
			supported: []string{oauth2.AuthMethodClientSecretPost, oauth2.AuthMethodClientSecretBasic},
			chosen:    oauth2.AuthMethodClientSecretPost,
		},
		{
			supported: []string{oauth2.AuthMethodClientSecretBasic, oauth2.AuthMethodClientSecretPost},
			chosen:    oauth2.AuthMethodClientSecretBasic,
		},
		{
			supported: []string{oauth2.AuthMethodClientSecretJWT, oauth2.AuthMethodClientSecretPost},
			chosen:    oauth2.AuthMethodClientSecretPost,
		},
		{
			supported: []string{oauth2.AuthMethodClientSecretJWT},
			chosen:    "",
			err:       true,
		},
	}

	for i, tt := range tests {
		cfg := ProviderConfig{
			TokenEndpointAuthMethodsSupported: tt.supported,
		}
		got, err := chooseAuthMethod(cfg)
		if tt.err {
			if err == nil {
				t.Errorf("case %d: expected non-nil err", i)
			}
			continue
		}

		if got != tt.chosen {
			t.Errorf("case %d: want=%q, got=%q", i, tt.chosen, got)
		}
	}
}

func TestClientMetadataUnmarshal(t *testing.T) {
	tests := []struct {
		data    string
		want    ClientMetadata
		wantErr bool
	}{
		{
			`{"redirect_uris":["https://example.com"]}`,
			ClientMetadata{
				RedirectURIs: []url.URL{
					{Scheme: "https", Host: "example.com"},
				},
			},
			false,
		},
		{
			// redirect_uris required
			`{}`,
			ClientMetadata{},
			true,
		},
		{
			// must have at least one redirect_uris
			`{"redirect_uris":[]}`,
			ClientMetadata{},
			true,
		},
		{
			`{"redirect_uris":["https://example.com"],"contacts":["Ms. Foo <foo@example.com>"]}`,
			ClientMetadata{
				RedirectURIs: []url.URL{
					{Scheme: "https", Host: "example.com"},
				},
				Contacts: []mail.Address{
					{Name: "Ms. Foo", Address: "foo@example.com"},
				},
			},
			false,
		},
		{
			// invalid URI provided for field
			`{"redirect_uris":["https://example.com"],"logo_uri":"not a valid uri"}`,
			ClientMetadata{},
			true,
		},
		{
			// logo_uri can't be a list
			`{"redirect_uris":["https://example.com"],"logo_uri":["https://example.com/logo"]}`,
			ClientMetadata{},
			true,
		},
		{
			`{
				"redirect_uris":["https://example.com"],
				"userinfo_encrypted_response_alg":"RSA1_5",
				"userinfo_encrypted_response_enc":"A128CBC-HS256",
				"contacts": [
					"jane doe <jane.doe@example.com>", "john doe <john.doe@example.com>"
				]
			}`,
			ClientMetadata{
				RedirectURIs: []url.URL{
					{Scheme: "https", Host: "example.com"},
				},
				UserInfoResponseOptions: JWAOptions{
					EncryptionAlg: "RSA1_5",
					EncryptionEnc: "A128CBC-HS256",
				},
				Contacts: []mail.Address{
					{Name: "jane doe", Address: "jane.doe@example.com"},
					{Name: "john doe", Address: "john.doe@example.com"},
				},
			},
			false,
		},
		{
			// If encrypted_response_enc is provided encrypted_response_alg must also be.
			`{
				"redirect_uris":["https://example.com"],
				"userinfo_encrypted_response_enc":"A128CBC-HS256"
			}`,
			ClientMetadata{},
			true,
		},
	}
	for i, tt := range tests {
		var got ClientMetadata
		if err := got.UnmarshalJSON([]byte(tt.data)); err != nil {
			if !tt.wantErr {
				t.Errorf("case %d: unmarshal failed: %v", i, err)
			}
			continue
		}
		if tt.wantErr {
			t.Errorf("case %d: expected unmarshal to produce error", i)
			continue
		}

		if diff := pretty.Compare(tt.want, got); diff != "" {
			t.Errorf("case %d: results not equal: %s", i, diff)
		}
	}
}

func TestClientMetadataMarshal(t *testing.T) {

	tests := []struct {
		metadata ClientMetadata
		want     string
	}{
		{
			ClientMetadata{
				RedirectURIs: []url.URL{
					{Scheme: "https", Host: "example.com", Path: "/callback"},
				},
			},
			`{"redirect_uris":["https://example.com/callback"]}`,
		},
		{
			ClientMetadata{
				RedirectURIs: []url.URL{
					{Scheme: "https", Host: "example.com", Path: "/callback"},
				},
				RequestObjectOptions: JWAOptions{
					EncryptionAlg: "RSA1_5",
					EncryptionEnc: "A128CBC-HS256",
				},
			},
			`{"redirect_uris":["https://example.com/callback"],"request_object_encryption_alg":"RSA1_5","request_object_encryption_enc":"A128CBC-HS256"}`,
		},
	}

	for i, tt := range tests {
		got, err := json.Marshal(&tt.metadata)
		if err != nil {
			t.Errorf("case %d: failed to marshal metadata: %v", i, err)
			continue
		}
		if string(got) != tt.want {
			t.Errorf("case %d: marshaled string did not match expected string", i)
		}
	}
}

func TestClientMetadataMarshalRoundTrip(t *testing.T) {
	tests := []ClientMetadata{
		{
			RedirectURIs: []url.URL{
				{Scheme: "https", Host: "example.com", Path: "/callback"},
			},
			LogoURI: &url.URL{Scheme: "https", Host: "example.com", Path: "/logo"},
			RequestObjectOptions: JWAOptions{
				EncryptionAlg: "RSA1_5",
				EncryptionEnc: "A128CBC-HS256",
			},
			ApplicationType:         "native",
			TokenEndpointAuthMethod: "client_secret_basic",
		},
	}

	for i, want := range tests {
		data, err := json.Marshal(&want)
		if err != nil {
			t.Errorf("case %d: failed to marshal metadata: %v", i, err)
			continue
		}
		var got ClientMetadata
		if err := json.Unmarshal(data, &got); err != nil {
			t.Errorf("case %d: failed to unmarshal metadata: %v", i, err)
			continue
		}
		if diff := pretty.Compare(want, got); diff != "" {
			t.Errorf("case %d: struct did not survive a marshaling round trip: %s", i, diff)
		}
	}
}

func TestClientRegistrationResponseUnmarshal(t *testing.T) {
	tests := []struct {
		data          string
		want          ClientRegistrationResponse
		wantErr       bool
		secretExpires bool
	}{
		{
			`{
				"client_id":"foo",
				"client_secret":"bar",
				"client_secret_expires_at": 1577858400,
				"redirect_uris":[
					"https://client.example.org/callback",
					"https://client.example.org/callback2"
				],
				"client_name":"my_example"
			}`,
			ClientRegistrationResponse{
				ClientID:              "foo",
				ClientSecret:          "bar",
				ClientSecretExpiresAt: time.Unix(1577858400, 0),
				ClientMetadata: ClientMetadata{
					RedirectURIs: []url.URL{
						{Scheme: "https", Host: "client.example.org", Path: "/callback"},
						{Scheme: "https", Host: "client.example.org", Path: "/callback2"},
					},
					ClientName: "my_example",
				},
			},
			false,
			true,
		},
		{
			`{
				"client_id":"foo",
				"client_secret_expires_at": 0,
				"redirect_uris":[
					"https://client.example.org/callback",
					"https://client.example.org/callback2"
				],
				"client_name":"my_example"
			}`,
			ClientRegistrationResponse{
				ClientID: "foo",
				ClientMetadata: ClientMetadata{
					RedirectURIs: []url.URL{
						{Scheme: "https", Host: "client.example.org", Path: "/callback"},
						{Scheme: "https", Host: "client.example.org", Path: "/callback2"},
					},
					ClientName: "my_example",
				},
			},
			false,
			false,
		},
		{
			// no client id
			`{
				"client_secret_expires_at": 0,
				"redirect_uris":[
					"https://client.example.org/callback",
					"https://client.example.org/callback2"
				],
				"client_name":"my_example"
			}`,
			ClientRegistrationResponse{},
			true,
			false,
		},
	}

	for i, tt := range tests {
		var got ClientRegistrationResponse
		if err := json.Unmarshal([]byte(tt.data), &got); err != nil {
			if !tt.wantErr {
				t.Errorf("case %d: unmarshal failed: %v", i, err)
			}
			continue
		}

		if tt.wantErr {
			t.Errorf("case %d: expected unmarshal to produce error", i)
			continue
		}

		if diff := pretty.Compare(tt.want, got); diff != "" {
			t.Errorf("case %d: results not equal: %s", i, diff)
		}
		if tt.secretExpires && got.ClientSecretExpiresAt.IsZero() {
			t.Errorf("case %d: expected client_secret to expire, but it doesn't", i)
		} else if !tt.secretExpires && !got.ClientSecretExpiresAt.IsZero() {
			t.Errorf("case %d: expected client_secret to not expire, but it does", i)
		}
	}
}

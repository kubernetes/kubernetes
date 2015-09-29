package oidc

import (
	"net/url"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/key"
	"github.com/coreos/go-oidc/oauth2"
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
		c *Client
		h bool
	}{
		// all ok
		{
			c: &Client{
				providerConfig: ProviderConfig{
					Issuer:    "http://example.com",
					ExpiresAt: now.Add(time.Hour),
				},
			},
			h: true,
		},
		// zero-value ProviderConfig.ExpiresAt
		{
			c: &Client{
				providerConfig: ProviderConfig{
					Issuer: "http://example.com",
				},
			},
			h: true,
		},
		// expired ProviderConfig
		{
			c: &Client{
				providerConfig: ProviderConfig{
					Issuer:    "http://example.com",
					ExpiresAt: now.Add(time.Hour * -1),
				},
			},
			h: false,
		},
		// empty ProviderConfig
		{
			c: &Client{},
			h: false,
		},
	}

	for i, tt := range tests {
		err := tt.c.Healthy()
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
			RedirectURLs: []url.URL{url.URL{Scheme: "http", Host: "example.com"}},
		},

		// one RedirectURL w/ nonempty path
		ClientMetadata{
			RedirectURLs: []url.URL{url.URL{Scheme: "http", Host: "example.com", Path: "/foo"}},
		},

		// two RedirectURLs
		ClientMetadata{
			RedirectURLs: []url.URL{
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
			RedirectURLs: nil,
		},

		// empty RedirectURLs slice
		ClientMetadata{
			RedirectURLs: []url.URL{},
		},

		// empty url.URL
		ClientMetadata{
			RedirectURLs: []url.URL{url.URL{}},
		},

		// empty url.URL following OK item
		ClientMetadata{
			RedirectURLs: []url.URL{url.URL{Scheme: "http", Host: "example.com"}, url.URL{}},
		},

		// url.URL with empty Host
		ClientMetadata{
			RedirectURLs: []url.URL{url.URL{Scheme: "http", Host: ""}},
		},

		// url.URL with empty Scheme
		ClientMetadata{
			RedirectURLs: []url.URL{url.URL{Scheme: "", Host: "example.com"}},
		},

		// url.URL with non-HTTP(S) Scheme
		ClientMetadata{
			RedirectURLs: []url.URL{url.URL{Scheme: "tcp", Host: "127.0.0.1"}},
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
		client := Client{
			providerConfig: ProviderConfig{
				TokenEndpointAuthMethodsSupported: tt.supported,
			},
		}
		got, err := client.chooseAuthMethod()
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

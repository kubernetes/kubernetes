package oidc

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"golang.org/x/net/context"
	jose "gopkg.in/square/go-jose.v2"
)

type keyServer struct {
	keys jose.JSONWebKeySet
}

func newKeyServer(keys ...jose.JSONWebKey) keyServer {
	return keyServer{
		keys: jose.JSONWebKeySet{Keys: keys},
	}
}

func (k keyServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if err := json.NewEncoder(w).Encode(k.keys); err != nil {
		panic(err)
	}
}

func TestKeysFormID(t *testing.T) {
	tests := []struct {
		name     string
		keys     []jose.JSONWebKey
		keyIDs   []string
		wantKeys []jose.JSONWebKey
	}{
		{
			name: "single key",
			keys: []jose.JSONWebKey{
				testKeyRSA_2048_0,
				testKeyECDSA_256_0,
			},
			keyIDs: []string{
				testKeyRSA_2048_0.KeyID,
			},
			wantKeys: []jose.JSONWebKey{
				testKeyRSA_2048_0,
			},
		},
		{
			name: "one key id matches",
			keys: []jose.JSONWebKey{
				testKeyRSA_2048_0,
				testKeyECDSA_256_0,
			},
			keyIDs: []string{
				testKeyRSA_2048_0.KeyID,
				testKeyRSA_2048_1.KeyID,
			},
			wantKeys: []jose.JSONWebKey{
				testKeyRSA_2048_0,
			},
		},
		{
			name: "no valid keys",
			keys: []jose.JSONWebKey{
				testKeyRSA_2048_1,
				testKeyECDSA_256_0,
			},
			keyIDs: []string{
				testKeyRSA_2048_0.KeyID,
			},
		},
	}

	t0 := time.Now()
	now := func() time.Time { return t0 }

	for _, test := range tests {
		func() {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			server := httptest.NewServer(newKeyServer(test.keys...))
			defer server.Close()

			keySet := newRemoteKeySet(ctx, server.URL, now)
			gotKeys, err := keySet.keysWithID(ctx, test.keyIDs)
			if err != nil {
				t.Errorf("%s: %v", test.name, err)
				return
			}
			if !reflect.DeepEqual(gotKeys, test.wantKeys) {
				t.Errorf("%s: expected keys=%#v, got=%#v", test.name, test.wantKeys, gotKeys)
			}
		}()
	}
}

package oidc

import (
	"testing"

	"k8s.io/kubernetes/pkg/util/diff"

	"github.com/coreos/go-oidc/jose"
)

func TestNewOIDCAuthProvider(t *testing.T) {
	tests := []struct {
		cfg map[string]string

		wantErr            bool
		wantInitialIDToken jose.JWT
	}{
		{
			cfg: map[string]string{
				cfgIssuerUrl: "auth.example.com",
			},
		},
	}

	for i, tt := range tests {
		ap, err := newOIDCAuthProvider("cluster.example.com", tt.cfg, nil)
		if tt.wantErr {
			if err == nil {
				t.Errorf("case %d: want non-nil err", i)
				continue
			}
		}

		if err != nil {
			t.Errorf("case %d: unexpected error on newOIDCAuthProvider: %v", i, err)
			continue
		}

		oidcAP, ok := ap.(*oidcAuthProvider)
		if !ok {
			t.Errorf("case %d: expected ap to be an oidcAuthProvider", i)
			continue
		}

		if diff := compareJWTs(tt.wantInitialIDToken, oidcAP.initialIDToken); diff != "" {
			t.Errorf("case %d: compareJWTs(tt.wantInitialIDToken, oidcAP.initialIDToken)=%v", i, diff)
		}
	}
}

func compareJWTs(a, b jose.JWT) string {
	if a.Encode() == b.Encode() {
		return ""
	}

	var aClaims, bClaims jose.Claims
	for _, j := range []struct {
		claims *jose.Claims
		jwt    jose.JWT
	}{
		{&aClaims, a},
		{&bClaims, b},
	} {
		var err error
		*j.claims, err = j.jwt.Claims()
		if err != nil {
			*j.claims = jose.Claims(map[string]interface{}{
				"msg": "bad claims",
				"err": err,
			})
		}
	}

	return diff.ObjectDiff(a, b)
}

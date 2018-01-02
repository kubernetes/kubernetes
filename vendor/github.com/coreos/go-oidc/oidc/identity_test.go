package oidc

import (
	"reflect"
	"testing"
	"time"

	"github.com/coreos/go-oidc/jose"
)

func TestIdentityFromClaims(t *testing.T) {
	tests := []struct {
		claims jose.Claims
		want   Identity
	}{
		{
			claims: jose.Claims{
				"sub":   "123850281",
				"name":  "Elroy",
				"email": "elroy@example.com",
				"exp":   float64(1.416935146e+09),
			},
			want: Identity{
				ID:        "123850281",
				Name:      "",
				Email:     "elroy@example.com",
				ExpiresAt: time.Date(2014, time.November, 25, 17, 05, 46, 0, time.UTC),
			},
		},
		{
			claims: jose.Claims{
				"sub":  "123850281",
				"name": "Elroy",
				"exp":  float64(1.416935146e+09),
			},
			want: Identity{
				ID:        "123850281",
				Name:      "",
				Email:     "",
				ExpiresAt: time.Date(2014, time.November, 25, 17, 05, 46, 0, time.UTC),
			},
		},
		{
			claims: jose.Claims{
				"sub":   "123850281",
				"name":  "Elroy",
				"email": "elroy@example.com",
				"exp":   int64(1416935146),
			},
			want: Identity{
				ID:        "123850281",
				Name:      "",
				Email:     "elroy@example.com",
				ExpiresAt: time.Date(2014, time.November, 25, 17, 05, 46, 0, time.UTC),
			},
		},
		{
			claims: jose.Claims{
				"sub":   "123850281",
				"name":  "Elroy",
				"email": "elroy@example.com",
			},
			want: Identity{
				ID:        "123850281",
				Name:      "",
				Email:     "elroy@example.com",
				ExpiresAt: time.Time{},
			},
		},
	}

	for i, tt := range tests {
		got, err := IdentityFromClaims(tt.claims)
		if err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
			continue
		}
		if !reflect.DeepEqual(tt.want, *got) {
			t.Errorf("case %d: want=%#v got=%#v", i, tt.want, *got)
		}
	}
}

func TestIdentityFromClaimsFail(t *testing.T) {
	tests := []jose.Claims{
		// sub incorrect type
		jose.Claims{
			"sub":   123,
			"name":  "foo",
			"email": "elroy@example.com",
		},
		// email incorrect type
		jose.Claims{
			"sub":   "123850281",
			"name":  "Elroy",
			"email": false,
		},
		// exp incorrect type
		jose.Claims{
			"sub":   "123850281",
			"name":  "Elroy",
			"email": "elroy@example.com",
			"exp":   "2014-11-25 18:05:46 +0000 UTC",
		},
	}

	for i, tt := range tests {
		_, err := IdentityFromClaims(tt)
		if err == nil {
			t.Errorf("case %d: expected non-nil error", i)
		}
	}
}

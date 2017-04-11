package oidc

import (
	"errors"
	"fmt"
	"time"

	"github.com/jonboulle/clockwork"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/go-oidc/key"
)

func VerifySignature(jwt jose.JWT, keys []key.PublicKey) (bool, error) {
	jwtBytes := []byte(jwt.Data())
	for _, k := range keys {
		v, err := k.Verifier()
		if err != nil {
			return false, err
		}
		if v.Verify(jwt.Signature, jwtBytes) == nil {
			return true, nil
		}
	}
	return false, nil
}

// containsString returns true if the given string(needle) is found
// in the string array(haystack).
func containsString(needle string, haystack []string) bool {
	for _, v := range haystack {
		if v == needle {
			return true
		}
	}
	return false
}

// Verify claims in accordance with OIDC spec
// http://openid.net/specs/openid-connect-basic-1_0.html#IDTokenValidation
func VerifyClaims(jwt jose.JWT, issuer, clientID string) error {
	now := time.Now().UTC()

	claims, err := jwt.Claims()
	if err != nil {
		return err
	}

	ident, err := IdentityFromClaims(claims)
	if err != nil {
		return err
	}

	if ident.ExpiresAt.Before(now) {
		return errors.New("token is expired")
	}

	// iss REQUIRED. Issuer Identifier for the Issuer of the response.
	// The iss value is a case sensitive URL using the https scheme that contains scheme,
	// host, and optionally, port number and path components and no query or fragment components.
	if iss, exists := claims["iss"].(string); exists {
		if !urlEqual(iss, issuer) {
			return fmt.Errorf("invalid claim value: 'iss'. expected=%s, found=%s.", issuer, iss)
		}
	} else {
		return errors.New("missing claim: 'iss'")
	}

	// iat REQUIRED. Time at which the JWT was issued.
	// Its value is a JSON number representing the number of seconds from 1970-01-01T0:0:0Z
	// as measured in UTC until the date/time.
	if _, exists := claims["iat"].(float64); !exists {
		return errors.New("missing claim: 'iat'")
	}

	// aud REQUIRED. Audience(s) that this ID Token is intended for.
	// It MUST contain the OAuth 2.0 client_id of the Relying Party as an audience value.
	// It MAY also contain identifiers for other audiences. In the general case, the aud
	// value is an array of case sensitive strings. In the common special case when there
	// is one audience, the aud value MAY be a single case sensitive string.
	if aud, ok, err := claims.StringClaim("aud"); err == nil && ok {
		if aud != clientID {
			return fmt.Errorf("invalid claims, 'aud' claim and 'client_id' do not match, aud=%s, client_id=%s", aud, clientID)
		}
	} else if aud, ok, err := claims.StringsClaim("aud"); err == nil && ok {
		if !containsString(clientID, aud) {
			return fmt.Errorf("invalid claims, cannot find 'client_id' in 'aud' claim, aud=%v, client_id=%s", aud, clientID)
		}
	} else {
		return errors.New("invalid claim value: 'aud' is required, and should be either string or string array")
	}

	return nil
}

// VerifyClientClaims verifies all the required claims are valid for a "client credentials" JWT.
// Returns the client ID if valid, or an error if invalid.
func VerifyClientClaims(jwt jose.JWT, issuer string) (string, error) {
	claims, err := jwt.Claims()
	if err != nil {
		return "", fmt.Errorf("failed to parse JWT claims: %v", err)
	}

	iss, ok, err := claims.StringClaim("iss")
	if err != nil {
		return "", fmt.Errorf("failed to parse 'iss' claim: %v", err)
	} else if !ok {
		return "", errors.New("missing required 'iss' claim")
	} else if !urlEqual(iss, issuer) {
		return "", fmt.Errorf("'iss' claim does not match expected issuer, iss=%s", iss)
	}

	sub, ok, err := claims.StringClaim("sub")
	if err != nil {
		return "", fmt.Errorf("failed to parse 'sub' claim: %v", err)
	} else if !ok {
		return "", errors.New("missing required 'sub' claim")
	}

	if aud, ok, err := claims.StringClaim("aud"); err == nil && ok {
		if aud != sub {
			return "", fmt.Errorf("invalid claims, 'aud' claim and 'sub' claim do not match, aud=%s, sub=%s", aud, sub)
		}
	} else if aud, ok, err := claims.StringsClaim("aud"); err == nil && ok {
		if !containsString(sub, aud) {
			return "", fmt.Errorf("invalid claims, cannot find 'sud' in 'aud' claim, aud=%v, sub=%s", aud, sub)
		}
	} else {
		return "", errors.New("invalid claim value: 'aud' is required, and should be either string or string array")
	}

	now := time.Now().UTC()
	exp, ok, err := claims.TimeClaim("exp")
	if err != nil {
		return "", fmt.Errorf("failed to parse 'exp' claim: %v", err)
	} else if !ok {
		return "", errors.New("missing required 'exp' claim")
	} else if exp.Before(now) {
		return "", fmt.Errorf("token already expired at: %v", exp)
	}

	return sub, nil
}

type JWTVerifier struct {
	issuer   string
	clientID string
	syncFunc func() error
	keysFunc func() []key.PublicKey
	clock    clockwork.Clock
}

func NewJWTVerifier(issuer, clientID string, syncFunc func() error, keysFunc func() []key.PublicKey) JWTVerifier {
	return JWTVerifier{
		issuer:   issuer,
		clientID: clientID,
		syncFunc: syncFunc,
		keysFunc: keysFunc,
		clock:    clockwork.NewRealClock(),
	}
}

func (v *JWTVerifier) Verify(jwt jose.JWT) error {
	// Verify claims before verifying the signature. This is an optimization to throw out
	// tokens we know are invalid without undergoing an expensive signature check and
	// possibly a re-sync event.
	if err := VerifyClaims(jwt, v.issuer, v.clientID); err != nil {
		return fmt.Errorf("oidc: JWT claims invalid: %v", err)
	}

	ok, err := VerifySignature(jwt, v.keysFunc())
	if err != nil {
		return fmt.Errorf("oidc: JWT signature verification failed: %v", err)
	} else if ok {
		return nil
	}

	if err = v.syncFunc(); err != nil {
		return fmt.Errorf("oidc: failed syncing KeySet: %v", err)
	}

	ok, err = VerifySignature(jwt, v.keysFunc())
	if err != nil {
		return fmt.Errorf("oidc: JWT signature verification failed: %v", err)
	} else if !ok {
		return errors.New("oidc: unable to verify JWT signature: no matching keys")
	}

	return nil
}

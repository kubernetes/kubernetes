package oidc

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"
)

// LogoutToken represents a verified token from a Back-Channel Logout. Use
// [IDTokenVerifier.VerifyLogout] within a POST handler to receive and validate
// a token.
//
// See the ./example/logout at the top-level of this repo for a full example
// application.
type LogoutToken struct {
	// The required token ID claim ("jti"). When processing this token for
	// logout, applications should validate that no recent token with the same
	// value has been processed.
	TokenID string
	// The unique identifier of the user that this logout token is for. This
	// will match the subject value of the ID Token.
	//
	// If not set, SessionID will be.
	Subject string

	// The "iss" claim. This is validated against the provider URL unless
	// explicitly skipped through SkipIssuerCheck.
	Issuer string
	// The Client ID this logout token is for. Validated against the config
	// unless SkipClientIDCheck is provided.
	Audience []string
	// When this token was issued.
	IssuedAt time.Time
	// When this token expires. Validated unless SkipExpiryCheck is provided.
	Expiry time.Time
	// Optional session ID claim ("sid").
	//
	// The exact semantics of session IDs vary between identity providers. Use
	// your provider's documentation to determine what this correlates to and
	// how it should be handled.
	SessionID string

	claims []byte
}

// Claims unmarshals the raw JSON payload of the Logout Token into a provided
// struct. This can be used to access field not exposed by the LogoutToken
// fields.
//
//	logoutToken, err := idTokenVerifier.VerifyLogout(ctx, rawLogoutToken)
//	if err != nil{
//		// ...
//	}
//	var claims struct {
//		TraceID string `json:"trace_id"`
//	}
//	if err := logoutToken.Claims(&claims); err != nil {
//		// ...
//	}
func (l *LogoutToken) Claims(v any) error {
	if l.claims == nil {
		return errors.New("oidc: claims not set")
	}
	return json.Unmarshal(l.claims, v)
}

// https://openid.net/specs/openid-connect-backchannel-1_0.html#LogoutToken
type logoutTokenJSON struct {
	Issuer   string   `json:"iss"`
	Subject  string   `json:"sub"`
	Audience audience `json:"aud"`
	Expiry   jsonTime `json:"exp"`
	IssuedAt jsonTime `json:"iat"`
	JTI      string   `json:"jti"`
	Events   struct {
		Logout json.RawMessage `json:"http://schemas.openid.net/event/backchannel-logout"`
	}
	SessionID string `json:"sid"`
	// Nonce is parsed as a raw message so its mere presence can be detected.
	// The spec requires logout tokens to not contain a nonce claim, regardless
	// of its value.
	Nonce json.RawMessage `json:"nonce"`
}

// VerifyLogout validates a back-channel logout token. Logout tokens are
// received by the relying party (this package) from the identity provider at a
// preconfigured "backchannel_logout_uri" through a POST. Then on certain
// events, such as RP-Initiated Logout, the identity provider will send a signed
// token indicating that sessions for a specific user should be terminated.
//
// To support back-channel logout within your app, register a POST endpoint and
// verify the token:
//
//	oidcConfig := &oidc.Config{
//		ClientID: clientID,
//	}
//	verifier := provider.Verifier(oidcConfig)
//
//	mux.HandleFunc("POST /logout", func(w http.ResponseWriter, r *http.Request) {
//		rawLogoutToken := r.PostFormValue("logout_token")
//		if rawLogoutToken == "" {
//			// ...
//		}
//		logoutToken, err := verifier.VerifyLogout(r.Context(), rawLogoutToken)
//		if err != nil {
//			// ...
//		}
//		// Use fields in the logoutToken to determine what sessions to
//		// terminate.
//
//	})
//
// Back-channel logout spec: https://openid.net/specs/openid-connect-backchannel-1_0.html
//
// RP-initiated logout spec: https://openid.net/specs/openid-connect-rpinitiated-1_0.html
func (v *IDTokenVerifier) VerifyLogout(ctx context.Context, rawLogoutToken string) (*LogoutToken, error) {
	payload, _, err := v.verifyJWT(ctx, rawLogoutToken)
	if err != nil {
		return nil, err
	}
	var logoutToken logoutTokenJSON
	if err := json.Unmarshal(payload, &logoutToken); err != nil {
		return nil, fmt.Errorf("oidc: parsing logout token payload: %v", err)
	}

	if len(logoutToken.Events.Logout) == 0 {
		return nil, fmt.Errorf("oidc: logout token missing required http://schemas.openid.net/event/backchannel-logout event")
	}

	// A logout token MUST NOT contain a nonce claim. This prevents an ID token
	// from being replayed as a logout token.
	if len(logoutToken.Nonce) != 0 {
		return nil, fmt.Errorf("oidc: logout token must not contain a 'nonce' claim")
	}

	t := &LogoutToken{
		Issuer:    logoutToken.Issuer,
		Subject:   logoutToken.Subject,
		Audience:  logoutToken.Audience,
		IssuedAt:  time.Time(logoutToken.IssuedAt),
		Expiry:    time.Time(logoutToken.Expiry),
		SessionID: logoutToken.SessionID,
		TokenID:   logoutToken.JTI,
		claims:    payload,
	}
	if t.TokenID == "" {
		return nil, fmt.Errorf("oidc: logout token missing required claim 'jti'")
	}
	if t.Expiry.IsZero() {
		return nil, fmt.Errorf("oidc: logout token must contain an 'exp' claim")
	}

	// A logout token MUST contain a 'sub' claim, a 'sid' claim, or both.
	if t.Subject == "" && t.SessionID == "" {
		return nil, fmt.Errorf("oidc: logout token must contain a 'sub' claim, a 'sid' claim, or both")
	}

	if !v.config.SkipIssuerCheck && t.Issuer != v.issuer {
		return nil, fmt.Errorf("oidc: logout token issued by a different provider, expected %q, got %q", v.issuer, t.Issuer)
	}

	if !v.config.SkipClientIDCheck {
		if v.config.ClientID != "" {
			if !slices.Contains(t.Audience, v.config.ClientID) {
				return nil, fmt.Errorf("oidc: expected logout token audience %q got %q", v.config.ClientID, t.Audience)
			}
		} else {
			return nil, fmt.Errorf("oidc: invalid configuration, clientID must be provided or SkipClientIDCheck must be set")
		}
	}

	// If a SkipExpiryCheck is false, make sure token is not expired.
	if !v.config.SkipExpiryCheck {
		now := time.Now
		if v.config.Now != nil {
			now = v.config.Now
		}
		nowTime := now()

		if t.Expiry.Before(nowTime) {
			return nil, &TokenExpiredError{Expiry: t.Expiry}
		}
	}
	return t, nil
}

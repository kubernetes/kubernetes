// Package oidc implements OpenID Connect client logic for the golang.org/x/oauth2 package.
//
// Construct a [Provider] and [oauth2.Config] through the identity provider's
// discovery document with [NewProvider], and construct an ID Token verifier
// with [Provider.Verifier]:
//
//	provider, err := oidc.NewProvider(ctx, "https://accounts.google.com")
//	if err != nil {
//	    // handle error
//	}
//
//	// Configure an OpenID Connect aware OAuth2 client.
//	oauth2Config := oauth2.Config{
//	    ClientID:     clientID,
//	    ClientSecret: clientSecret,
//	    RedirectURL:  redirectURL,
//	    // Discovery returns the OAuth2 endpoints.
//	    Endpoint: provider.Endpoint(),
//	    // "openid" is a required scope for OpenID Connect flows.
//	    Scopes: []string{oidc.ScopeOpenID, oidc.ScopeProfile, oidc.ScopeEmail},
//	}
//
//	idTokenVerifier := provider.Verifier(&oidc.Config{ClientID: clientID})
//
// OAuth 2.0 redirects then opt into the OpenID Connect flow with [ScopeOpenID]:
//
//	func handleRedirect(w http.ResponseWriter, r *http.Request) {
//		http.Redirect(w, r, oauth2Config.AuthCodeURL(state), http.StatusFound)
//	}
//
// When handling an OAuth 2.0 response, an [IDTokenVerifier] can be used to
// validate the "id_token" payload, containing well-known fields such as the
// user's email, name, and picture URL:
//
//	func handleOAuth2Callback(w http.ResponseWriter, r *http.Request) {
//		// Verify state and other OAuth 2.0 responses.
//
//		// Perform standard token exchange.
//		oauth2Token, err := oauth2Config.Exchange(r.Context(), r.URL.Query().Get("code"))
//		if err != nil {
//			// ...
//		}
//		// Extract the ID Token from OAuth2 token.
//		rawIDToken, ok := oauth2Token.Extra("id_token").(string)
//		if !ok {
//			// ...
//		}
//		// Parse and verify ID Token payload.
//		idToken, err := idTokenVerifier.Verify(r.Context(), rawIDToken)
//		if err != nil {
//			// ...
//		}
//		// Parse well-known claims from the token.
//		// https://openid.net/specs/openid-connect-core-1_0.html#Claims
//		var claims struct {
//			Email         string `json:"email"`
//			EmailVerified bool   `json:"email_verified"`
//			Name          string `json:"name"`
//			Picture       string `json:"picture"`
//		}
//		if err := idToken.Claims(&claims); err != nil {
//			// ...
//		}
//	}
package oidc

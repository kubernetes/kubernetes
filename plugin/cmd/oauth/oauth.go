/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package oauth

import (
	"net/http"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/apiserver"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator/bearertoken"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"

	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/password/allow"
	sessionauth "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/request/session"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/request/union"
	accesstokenauth "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/token/accesstoken"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/server/csrf"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/server/grant"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/server/handlers"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/server/login"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/server/session"

	_ "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
	_ "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api/v1beta1"
	_ "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api/v1beta2"
	_ "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api/v1beta3"
	oauthclient "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/client"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/osinserver"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/osinserver/access"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/osinserver/authorize"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/osinserver/registrystorage"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/registry/accesstoken"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/registry/authorizetoken"
	oauthclientregistry "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/registry/client"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/registry/clientauthorization"

	"github.com/golang/glog"
)

const (
	OAuthPrefix   = "/oauth"
	ApprovePrefix = "/oauth/approve"
	LoginPrefix   = "/login"
)

// mux is an object that can register http handlers.
type Mux interface {
	Handle(pattern string, handler http.Handler)
	HandleFunc(pattern string, handler func(http.ResponseWriter, *http.Request))
}

func InstallAuthenticator(restClient *client.RESTClient, existingAuth authenticator.Request) authenticator.Request {
	tokenAuth := bearertoken.New(accesstokenauth.NewTokenAuthenticator(oauthclient.New(restClient)))
	if existingAuth == nil {
		return tokenAuth
	}
	return union.New(existingAuth, tokenAuth)
}

func InstallStorage(storage map[string]apiserver.RESTStorage, helper tools.EtcdHelper) {
	storage[oauthclient.OAuthAccessTokensPath] = accesstoken.NewREST(accesstoken.NewEtcdRegistry(helper))
	storage[oauthclient.OAuthAuthorizeTokensPath] = authorizetoken.NewREST(authorizetoken.NewEtcdRegistry(helper))
	storage[oauthclient.OAuthClientsPath] = oauthclientregistry.NewREST(oauthclientregistry.NewEtcdRegistry(helper))
	storage[oauthclient.OAuthClientAuthorizationsPath] = clientauthorization.NewREST(clientauthorization.NewEtcdRegistry(helper))
}

// Install starts an OAuth2 server and registers the OAuth and Login handlers
// into the provided mux, then returns an array of strings indicating what
// endpoints were started.
func InstallSupport(restClient *client.RESTClient, sessionSecrets []string, mux Mux) {

	oauthClient := oauthclient.New(restClient)

	// Chain of auth success handlers
	var authSuccessHandlers handlers.AuthenticationSuccessHandlers

	// Session-based request authenticator
	sessionStore := session.NewStore(sessionSecrets...)
	sessionAuth := sessionauth.New(sessionStore, "ssn", sessionauth.NewDefaultUserConversion())
	authSuccessHandlers = append(authSuccessHandlers, sessionAuth)

	// If auth fails, redirect to a login page and redirect back on login
	authNeededRedirect := authorize.NewRedirectAuthenticator(LoginPrefix, "then")
	authSuccessHandlers = append(authSuccessHandlers, handlers.RedirectSuccessHandler{})

	// Login page (cookie-based CSRF, stub password authenticator)
	loginServer := login.NewLogin(
		csrf.NewCookieCSRF("csrf", http.Cookie{}),
		allow.NewAllow(),
		authSuccessHandlers,
		login.DefaultLoginFormRenderer)
	loginServer.Install(mux, LoginPrefix)

	// Check requested grants against stored authorizations
	grantChecker := registrystorage.NewClientAuthorizationGrantChecker(oauthClient)

	// If an unauthorized grant is requested, redirect to a grant approval page
	grantHandler := authorize.NewRedirectGrant(ApprovePrefix)

	// Grant approval page (cookie-based CSRF)
	grantServer := grant.NewGrant(
		csrf.NewCookieCSRF("csrf", http.Cookie{}),
		sessionAuth,
		grant.DefaultGrantFormRenderer,
		oauthClient)
	grantServer.Install(mux, ApprovePrefix)

	// OAuth server
	oauthServer := osinserver.New(
		osinserver.NewDefaultServerConfig(),
		registrystorage.New(oauthClient, registrystorage.NewUserConversion()),
		osinserver.AuthorizeHandlers{
			// Step 1: Authenticate the request
			authorize.NewAuthenticator(
				// Auth by checking session
				sessionAuth,
				// Redirect unauthenticated requests to login page
				authNeededRedirect,
			),
			// Step 2: Ensure requesting client is approved
			authorize.NewGrantCheck(
				// Check grants against stored client authorizations
				grantChecker,
				// Redirect to grant approval page if needed
				grantHandler,
			),
		},
		osinserver.AccessHandlers{
			// Deny all non-token access requests
			access.NewDenyAuthenticator(),
		},
		// Handle errors by printing the error message with a http error status
		osinserver.NewDefaultErrorHandler(),
	)
	oauthServer.Install(mux, OAuthPrefix)

	glog.Infof("Started OAuth2 server at %s", OAuthPrefix)
	glog.Infof("Started Login server at %s", LoginPrefix)
}

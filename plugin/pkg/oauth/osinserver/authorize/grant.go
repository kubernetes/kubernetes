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

package authorize

import (
	"errors"
	"net/http"
	"net/url"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth"
	oapi "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
	oauthclient "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/client"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/scope"
	"github.com/RangelReale/osin"
	"github.com/golang/glog"
)

// GrantChecker is responsible for determining if a user has authorized a requested grant
type GrantChecker interface {
	// HasAuthorizedClient returns true if the user has authorized the requested grant
	HasAuthorizedClient(user user.Info, grant oauth.Grant) (bool, error)
}

// GrantHandler handles errors during the grant process, or the client requests an unauthorized grant
type GrantHandler interface {
	// GrantNeeded reacts when a client requests an unauthorized grant
	// Returns true if the request was handled
	// Returns false if the request was not handled
	GrantNeeded(user user.Info, grant oauth.Grant, w http.ResponseWriter, req *http.Request) (bool, error)
}

// GrantCheck implements osinserver.AuthorizeHandler to ensure requested scopes have been authorized
type GrantCheck struct {
	check   GrantChecker
	handler GrantHandler
}

// NewGrantCheck returns a new GrantCheck
func NewGrantCheck(check GrantChecker, handler GrantHandler) *GrantCheck {
	return &GrantCheck{check, handler}
}

// HandleAuthorize implements osinserver.AuthorizeHandler to ensure the requested scopes have been authorized.
// If the requested scopes are authorized, the AuthorizeRequest is unchanged.
// If the requested scopes are not authorized, the request is handled.
func (h *GrantCheck) HandleAuthorize(ar *osin.AuthorizeRequest, w http.ResponseWriter) (handled bool, err error) {
	if !ar.Authorized {
		// Previously unauthorized requests remain unauthorized
		return false, nil
	}

	user, ok := ar.UserData.(user.Info)
	if !ok || user == nil {
		return false, errors.New("the provided user data is not user.Info")
	}

	grant := &oauth.DefaultGrant{
		Client:      ar.Client,
		Scopes:      scope.Split(ar.Scope),
		Expiration:  int64(ar.Expiration),
		RedirectURI: ar.RedirectUri,
	}

	ok, err = h.check.HasAuthorizedClient(user, grant)
	if err != nil {
		// The request is now considered unauthorized
		ar.Authorized = false
		return false, err
	}
	if !ok {
		// The request is now considered unauthorized
		ar.Authorized = false
		return h.handler.GrantNeeded(user, grant, w, ar.HttpRequest)
	}

	return false, nil
}

// emptyGrant is a no-op grant handler
type emptyGrant struct{}

// NewEmptyGrant returns a no-op grant handler
func NewEmptyGrant() GrantHandler {
	return emptyGrant{}
}

// GrantNeeded implements the GrantHandler interface
func (emptyGrant) GrantNeeded(user user.Info, grant oauth.Grant, w http.ResponseWriter, req *http.Request) (handled bool, err error) {
	glog.Infof("GrantNeeded: %#v\n%#v\n%#v", user, grant)
	return false, nil
}

// autoGrant automatically creates a client authorization when requested
type autoGrant struct {
	client oauthclient.Client
}

// NewAutoGrant returns a grant handler that automatically creates client authorizations
// when a grant is needed, then retries the original request
func NewAutoGrant(client oauthclient.Client) GrantHandler {
	return &autoGrant{client}
}

// GrantNeeded implements the GrantHandler interface
func (g *autoGrant) GrantNeeded(user user.Info, grant oauth.Grant, w http.ResponseWriter, req *http.Request) (handled bool, err error) {
	clientAuthID := g.client.OAuthClientAuthorizations().Name(user.GetName(), grant.GetClient().GetId())
	clientAuth, err := g.client.OAuthClientAuthorizations().Get(clientAuthID)
	if err == nil {
		// Add new scopes and update
		clientAuth.Scopes = scope.Add(clientAuth.Scopes, grant.GetScopes())
		if _, err = g.client.OAuthClientAuthorizations().Update(clientAuth); err != nil {
			glog.Errorf("Unable to update authorization: %v", err)
			return false, err
		}
	} else {
		clientAuth = &oapi.OAuthClientAuthorization{
			UserName:   user.GetName(),
			UserUID:    user.GetUID(),
			ClientName: grant.GetClient().GetId(),
			Scopes:     grant.GetScopes(),
		}
		clientAuth.Name = clientAuthID
		if _, err = g.client.OAuthClientAuthorizations().Create(clientAuth); err != nil {
			glog.Errorf("Unable to create authorization: %v", err)
			return false, err
		}
	}

	// Retry the request
	http.Redirect(w, req, req.URL.String(), http.StatusFound)
	return true, nil
}

// redirectGrant redirects grant requests to a URL
type redirectGrant struct {
	url string
}

// NewRedirectGrant returns a grant handler that redirects to the given URL when a grant is needed.
// The following query parameters are added to the URL:
//   then - original request URL
//   client_id - requesting client's ID
//   scopes - grant scope requested
//   redirect_uri - original authorize request redirect_uri
func NewRedirectGrant(url string) GrantHandler {
	return &redirectGrant{url}
}

// GrantNeeded implements the GrantHandler interface
func (g *redirectGrant) GrantNeeded(user user.Info, grant oauth.Grant, w http.ResponseWriter, req *http.Request) (handled bool, err error) {
	redirectURL, err := url.Parse(g.url)
	if err != nil {
		return false, err
	}
	redirectURL.RawQuery = url.Values{
		"then":         {req.URL.String()},
		"client_id":    {grant.GetClient().GetId()},
		"scopes":       grant.GetScopes(),
		"redirect_uri": {grant.GetRedirectUri()},
	}.Encode()
	http.Redirect(w, req, redirectURL.String(), http.StatusFound)
	return true, nil
}

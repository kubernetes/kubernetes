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

package grant

import (
	"html/template"
	"net/http"
	"net/url"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/server/csrf"
	oapi "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/api"
	oauthclient "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/client"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/oauth/scope"
	"github.com/golang/glog"
)

const (
	thenParam        = "then"
	csrfParam        = "csrf"
	clientIDParam    = "client_id"
	userNameParam    = "user_name"
	scopesParam      = "scopes"
	redirectURIParam = "redirect_uri"

	approveParam = "approve"
	denyParam    = "deny"
)

// GrantFormRenderer is responsible for rendering a GrantForm to prompt the user
// to approve or reject a requested OAuth scope grant.
type GrantFormRenderer interface {
	Render(form GrantForm, w http.ResponseWriter, req *http.Request)
}

type GrantForm struct {
	Action string
	Error  string
	Values GrantFormValues
}

type GrantFormValues struct {
	Then             string
	ThenParam        string
	CSRF             string
	CSRFParam        string
	ClientID         string
	ClientIDParam    string
	UserName         string
	UserNameParam    string
	Scopes           string
	ScopesParam      string
	RedirectURI      string
	RedirectURIParam string
	ApproveParam     string
	DenyParam        string
}

type Grant struct {
	auth   authenticator.Request
	csrf   csrf.CSRF
	render GrantFormRenderer
	client oauthclient.Interface
}

func NewGrant(csrf csrf.CSRF, auth authenticator.Request, render GrantFormRenderer, client oauthclient.Interface) *Grant {
	return &Grant{
		auth:   auth,
		csrf:   csrf,
		render: render,
		client: client,
	}
}

// Install registers the grant handler into a mux. It is expected that the
// provided prefix will serve all operations. Path MUST NOT end in a slash.
func (l *Grant) Install(mux Mux, paths ...string) {
	for _, path := range paths {
		path = strings.TrimRight(path, "/")
		mux.HandleFunc(path, l.ServeHTTP)
	}
}

func (l *Grant) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	user, ok, err := l.auth.AuthenticateRequest(req)
	if err != nil || !ok {
		l.redirect("You must reauthenticate before continuing", w, req)
		return
	}

	clientID := req.FormValue(clientIDParam)
	client, err := l.client.OAuthClients().Get(clientID)
	if err != nil || client == nil {
		l.failed("Could not find client for client_id", w, req)
		return
	}

	switch req.Method {
	case "GET":
		l.handleGrantForm(user, client, w, req)
	case "POST":
		l.handleGrant(user, client, w, req)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (l *Grant) handleGrantForm(user user.Info, client *oapi.OAuthClient, w http.ResponseWriter, req *http.Request) {
	q := req.URL.Query()
	then := q.Get(thenParam)
	scopes := q.Get(scopesParam)
	redirectURI := q.Get(redirectURIParam)

	uri, err := getBaseURL(req)
	if err != nil {
		glog.Errorf("Unable to generate base URL: %v", err)
		http.Error(w, "Unable to determine URL", http.StatusInternalServerError)
		return
	}

	csrf, err := l.csrf.Generate(w, req)
	if err != nil {
		glog.Errorf("Unable to generate CSRF token: %v", err)
		l.failed("Could not generate CSRF token", w, req)
		return
	}

	form := GrantForm{
		Action: uri.String(),
		Values: GrantFormValues{
			Then:             then,
			ThenParam:        thenParam,
			CSRF:             csrf,
			CSRFParam:        csrfParam,
			ClientID:         client.Name,
			ClientIDParam:    clientIDParam,
			UserName:         user.GetName(),
			UserNameParam:    userNameParam,
			Scopes:           scopes,
			ScopesParam:      scopesParam,
			RedirectURI:      redirectURI,
			RedirectURIParam: redirectURIParam,
			ApproveParam:     approveParam,
			DenyParam:        denyParam,
		},
	}

	l.render.Render(form, w, req)
}

func (l *Grant) handleGrant(user user.Info, client *oapi.OAuthClient, w http.ResponseWriter, req *http.Request) {
	if ok, err := l.csrf.Check(req, req.FormValue(csrfParam)); !ok || err != nil {
		glog.Errorf("Unable to check CSRF token: %v", err)
		l.failed("Invalid CSRF token", w, req)
		return
	}

	then := req.FormValue(thenParam)
	scopes := req.FormValue(scopesParam)

	if len(req.FormValue(approveParam)) == 0 {
		// Redirect with rejection param
		url, err := url.Parse(then)
		if len(then) == 0 || err != nil {
			l.failed("Access denied, but no redirect URL was specified", w, req)
			return
		}
		q := url.Query()
		q["error"] = []string{"grant_denied"}
		url.RawQuery = q.Encode()
		http.Redirect(w, req, url.String(), http.StatusFound)
		return
	}

	clientAuthID := l.client.OAuthClientAuthorizations().Name(user.GetName(), client.Name)
	clientAuth, err := l.client.OAuthClientAuthorizations().Get(clientAuthID)
	if err == nil && clientAuth != nil {
		// Add new scopes and update
		clientAuth.Scopes = scope.Add(clientAuth.Scopes, scope.Split(scopes))
		if _, err = l.client.OAuthClientAuthorizations().Update(clientAuth); err != nil {
			glog.Errorf("Unable to update authorization: %v", err)
			l.failed("Could not update client authorization", w, req)
			return
		}
	} else {
		// Make sure client name, user name, grant scope, expiration, and redirect uri match
		clientAuth = &oapi.OAuthClientAuthorization{
			UserName:   user.GetName(),
			UserUID:    user.GetUID(),
			ClientName: client.Name,
			Scopes:     scope.Split(scopes),
		}
		clientAuth.Name = clientAuthID

		if _, err = l.client.OAuthClientAuthorizations().Create(clientAuth); err != nil {
			glog.Errorf("Unable to create authorization: %v", err)
			l.failed("Could not create client authorization", w, req)
			return
		}
	}

	if len(then) == 0 {
		l.failed("Access granted, but no redirect URL was specified", w, req)
		return
	}

	http.Redirect(w, req, then, http.StatusFound)
}

func (l *Grant) failed(reason string, w http.ResponseWriter, req *http.Request) {
	form := GrantForm{
		Error: reason,
	}
	l.render.Render(form, w, req)
}
func (l *Grant) redirect(reason string, w http.ResponseWriter, req *http.Request) {
	then := req.FormValue("then")

	// TODO: validate then
	if len(then) == 0 {
		l.failed(reason, w, req)
		return
	}
	http.Redirect(w, req, then, http.StatusFound)
}

func getBaseURL(req *http.Request) (*url.URL, error) {
	uri, err := url.Parse(req.RequestURI)
	if err != nil {
		return nil, err
	}
	uri.Scheme, uri.Host, uri.RawQuery, uri.Fragment = req.URL.Scheme, req.URL.Host, "", ""
	return uri, nil
}

// DefaultGrantFormRenderer displays a page prompting the user to approve an OAuth grant.
// The requesting client id, requested scopes, and redirect URI are displayed to the user.
var DefaultGrantFormRenderer = grantTemplateRenderer{}

type grantTemplateRenderer struct{}

func (r grantTemplateRenderer) Render(form GrantForm, w http.ResponseWriter, req *http.Request) {
	w.Header().Add("Content-Type", "text/html")
	w.WriteHeader(http.StatusOK)
	if err := grantTemplate.Execute(w, form); err != nil {
		glog.Errorf("Unable to render grant template: %v", err)
	}
}

var grantTemplate = template.Must(template.New("grantForm").Parse(`
{{ if .Error }}
<div class="message">{{ .Error }}</div>
{{ else }}
<form action="{{ .Action }}" method="POST">
  <input type="hidden" name="{{ .Values.ThenParam }}" value="{{ .Values.Then }}">
  <input type="hidden" name="{{ .Values.CSRFParam }}" value="{{ .Values.CSRF }}">
  <input type="hidden" name="{{ .Values.ClientIDParam }}" value="{{ .Values.ClientID }}">
  <input type="hidden" name="{{ .Values.UserNameParam }}" value="{{ .Values.UserName }}">
  <input type="hidden" name="{{ .Values.ScopesParam }}" value="{{ .Values.Scopes }}">
  <input type="hidden" name="{{ .Values.RedirectURIParam }}" value="{{ .Values.RedirectURI }}">

  <div>Do you approve this client?</div>
  <div>Client:     {{ .Values.ClientID }}</div>
  <div>Scope:      {{ .Values.Scopes }}</div>
  <div>URI:        {{ .Values.RedirectURI }}</div>
  
  <input type="submit" name="{{ .Values.ApproveParam }}" value="Approve">
  <input type="submit" name="{{ .Values.DenyParam }}" value="Reject">
</form>

{{ end }}
`))

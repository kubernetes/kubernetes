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

package login

import (
	"html/template"
	"net/http"
	"net/url"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/server/csrf"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/server/handlers"
	"github.com/golang/glog"
)

type LoginFormRenderer interface {
	Render(form LoginForm, w http.ResponseWriter, req *http.Request)
}

type LoginForm struct {
	Action string
	Error  string
	Values LoginFormValues
}

type LoginFormValues struct {
	Then     string
	CSRF     string
	Username string
	Password string
}

type Login struct {
	csrf    csrf.CSRF
	auth    authenticator.Password
	success handlers.AuthenticationSuccessHandler
	render  LoginFormRenderer
}

func NewLogin(csrf csrf.CSRF, auth authenticator.Password, success handlers.AuthenticationSuccessHandler, render LoginFormRenderer) *Login {
	return &Login{
		csrf:    csrf,
		auth:    auth,
		success: success,
		render:  render,
	}
}

// Install registers the login handler into a mux. It is expected that the
// provided prefix will serve all operations. Path MUST NOT end in a slash.
func (l *Login) Install(mux Mux, paths ...string) {
	for _, path := range paths {
		path = strings.TrimRight(path, "/")
		mux.HandleFunc(path, l.ServeHTTP)
	}
}

func (l *Login) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	switch req.Method {
	case "GET":
		l.handleLoginForm(w, req)
	case "POST":
		l.handleLogin(w, req)
	default:
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
	}
}

func (l *Login) handleLoginForm(w http.ResponseWriter, req *http.Request) {
	uri, err := getBaseURL(req)
	if err != nil {
		glog.Errorf("Unable to generate base URL: %v", err)
		http.Error(w, "Unable to determine URL", http.StatusInternalServerError)
		return
	}

	form := LoginForm{
		Action: uri.String(),
	}
	if then := req.URL.Query().Get("then"); then != "" {
		// TODO: sanitize 'then'
		form.Values.Then = then
	}
	switch req.URL.Query().Get("reason") {
	case "":
		break
	case "user required":
		form.Error = "Login is required. Please try again."
	case "token expired":
		form.Error = "Could not check CSRF token. Please try again."
	case "access denied":
		form.Error = "Invalid login or password. Please try again."
	default:
		form.Error = "An unknown error has occured. Please try again."
	}

	csrf, err := l.csrf.Generate(w, req)
	if err != nil {
		glog.Errorf("Unable to generate CSRF token: %v", err)
	}
	form.Values.CSRF = csrf

	l.render.Render(form, w, req)
}

func (l *Login) handleLogin(w http.ResponseWriter, req *http.Request) {
	if ok, err := l.csrf.Check(req, req.FormValue("csrf")); !ok || err != nil {
		glog.Errorf("Unable to check CSRF token: %v", err)
		failed("token expired", w, req)
		return
	}
	then := req.FormValue("then")
	user, password := req.FormValue("username"), req.FormValue("password")
	if user == "" {
		failed("user required", w, req)
		return
	}
	context, ok, err := l.auth.AuthenticatePassword(user, password)
	if err != nil {
		glog.Errorf("Unable to authenticate password: %v", err)
		failed("unknown error", w, req)
		return
	}
	if !ok {
		failed("access denied", w, req)
		return
	}
	l.success.AuthenticationSucceeded(context, then, w, req)
}

func failed(reason string, w http.ResponseWriter, req *http.Request) {
	uri, err := getBaseURL(req)
	if err != nil {
		http.Redirect(w, req, req.URL.Path, http.StatusFound)
		return
	}
	query := url.Values{}
	query.Set("reason", reason)
	if then := req.FormValue("then"); then != "" {
		query.Set("then", then)
	}
	uri.RawQuery = query.Encode()
	http.Redirect(w, req, uri.String(), http.StatusFound)
}

func getBaseURL(req *http.Request) (*url.URL, error) {
	uri, err := url.Parse(req.RequestURI)
	if err != nil {
		return nil, err
	}
	uri.Scheme, uri.Host, uri.RawQuery, uri.Fragment = req.URL.Scheme, req.URL.Host, "", ""
	return uri, nil
}

var DefaultLoginFormRenderer = loginTemplateRenderer{}

type loginTemplateRenderer struct{}

func (r loginTemplateRenderer) Render(form LoginForm, w http.ResponseWriter, req *http.Request) {
	w.Header().Add("Content-Type", "text/html")
	w.WriteHeader(http.StatusOK)
	if err := loginTemplate.Execute(w, form); err != nil {
		glog.Errorf("Unable to render login template: %v", err)
	}
}

var loginTemplate = template.Must(template.New("loginForm").Parse(`
{{ if .Error }}<div class="message">{{ .Error }}</div>{{ end }}
<form action="{{ .Action }}" method="POST">
  <input type="hidden" name="then" value="{{ .Values.Then }}">
  <input type="hidden" name="csrf" value="{{ .Values.CSRF }}">
  <label>Login: <input type="text" name="username" value="{{ .Values.Username }}"></label>
  <label>Password: <input type="password" name="password" value=""></label>
  <input type="submit" value="Login">
</form>
`))

/*
Copyright 2016 The Kubernetes Authors.

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

package oidc

import (
	"context"
	"crypto/rand"
	"encoding/base64"
	"net"
	"net/http"
	"strings"

	oidcp "github.com/coreos/go-oidc"
	"github.com/skratchdot/open-golang/open"
	"golang.org/x/oauth2"
)

const (
	defaultCallbackPort = "8172"
	defaultCallbackPath = "/authcode"

	successPage = `<!DOCTYPE html>
<html>
<head><title>Success</title></head>
<body onload='window.close()'><p>Login successfully! Please close this browser.</p></body>
</html>
`
)

func (p *oidcAuthProvider) Login() error {
	issuer := p.cfg[cfgIssuerUrl]
	clientID := p.cfg[cfgClientID]
	callbackPort := p.cfg[cfgCallbackPort]
	if callbackPort == "" {
		callbackPort = defaultCallbackPort
	}
	callbackPath := "/" + strings.TrimLeft(p.cfg[cfgCallbackPath], "/")
	if callbackPath == "/" {
		callbackPath = defaultCallbackPath
	}
	scopes := []string{oidcp.ScopeOpenID}
	for _, scope := range strings.Split(p.cfg[cfgExtraScopes], ",") {
		if scope != "" {
			scopes = append(scopes, scope)
		}
	}

	options := []oauth2.AuthCodeOption{oauth2.AccessTypeOffline}
	for _, opt := range strings.Split(p.cfg[cfgAuthOptions], ";") {
		if pos := strings.Index(opt, "="); pos > 0 {
			options = append(options, oauth2.SetAuthURLParam(opt[:pos], opt[pos+1:]))
		}
	}

	ctx := context.WithValue(context.TODO(), oauth2.HTTPClient, p.client)
	provider, err := oidcp.NewProvider(ctx, issuer)
	if err != nil {
		return err
	}
	oidcConf := &oidcp.Config{ClientID: clientID}
	verifier := provider.Verifier(oidcConf)
	oauth2Conf := oauth2.Config{
		ClientID:     clientID,
		ClientSecret: p.cfg[cfgClientSecret],
		Endpoint:     provider.Endpoint(),
		RedirectURL:  "http://localhost:" + callbackPort + callbackPath,
		Scopes:       scopes,
	}

	state, err := generateState()
	if err != nil {
		return err
	}

	shutdownCh := make(chan struct{})

	mux := http.NewServeMux()
	server := &http.Server{Addr: "127.0.0.1:" + callbackPort, Handler: mux}

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, oauth2Conf.AuthCodeURL(state, options...), http.StatusFound)
	})

	mux.HandleFunc(callbackPath, func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("state") != state {
			http.Error(w, "invalid session", http.StatusBadRequest)
			return
		}
		token, err := oauth2Conf.Exchange(ctx, r.URL.Query().Get("code"))
		if err != nil {
			http.Error(w, "failed to exchange token: "+err.Error(), http.StatusInternalServerError)
			return
		}
		idToken, ok := token.Extra("id_token").(string)
		if !ok {
			http.Error(w, "id_token not available", http.StatusBadRequest)
			return
		}

		if _, err = verifier.Verify(ctx, idToken); err != nil {
			http.Error(w, "id_token verification failed: "+err.Error(), http.StatusBadRequest)
			return
		}

		tokens := make(map[string]string)
		for key, val := range p.cfg {
			tokens[key] = val
		}
		tokens[cfgIDToken] = idToken
		if token.RefreshToken != "" {
			tokens[cfgRefreshToken] = token.RefreshToken
		}
		if err = p.persister.Persist(tokens); err != nil {
			http.Error(w, "persist tokens failed: "+err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Add("Content-type", "text/html")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(successPage))

		go func() {
			server.Shutdown(context.Background())
			close(shutdownCh)
		}()
	})

	listener, err := net.Listen("tcp", server.Addr)
	if err != nil {
		return err
	}
	go open.Start("http://localhost:" + callbackPort)
	if err = server.Serve(listener); err != http.ErrServerClosed {
		return err
	}
	<-shutdownCh
	return nil
}

func generateState() (string, error) {
	buf := make([]byte, 8)
	if _, err := rand.Read(buf); err != nil {
		return "", err
	}
	return base64.URLEncoding.EncodeToString(buf), nil
}

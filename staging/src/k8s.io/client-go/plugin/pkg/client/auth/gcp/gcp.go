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

package gcp

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os/exec"
	"strings"
	"time"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"k8s.io/client-go/pkg/util/jsonpath"
	"k8s.io/client-go/pkg/util/yaml"
	"k8s.io/client-go/rest"
)

func init() {
	if err := rest.RegisterAuthProviderPlugin("gcp", newGCPAuthProvider); err != nil {
		glog.Fatalf("Failed to register gcp auth plugin: %v", err)
	}
}

type gcpAuthProvider struct {
	tokenSource oauth2.TokenSource
	persister   rest.AuthProviderConfigPersister
}

func newGCPAuthProvider(_ string, gcpConfig map[string]string, persister rest.AuthProviderConfigPersister) (rest.AuthProvider, error) {
	cmd, useCmd := gcpConfig["cmd-path"]
	var ts oauth2.TokenSource
	var err error
	if useCmd {
		ts, err = newCmdTokenSource(cmd, gcpConfig["token-key"], gcpConfig["expiry-key"], gcpConfig["time-fmt"])
	} else {
		ts, err = google.DefaultTokenSource(context.Background(), "https://www.googleapis.com/auth/cloud-platform")
	}
	if err != nil {
		return nil, err
	}
	cts, err := newCachedTokenSource(gcpConfig["access-token"], gcpConfig["expiry"], persister, ts, gcpConfig)
	if err != nil {
		return nil, err
	}
	return &gcpAuthProvider{cts, persister}, nil
}

func (g *gcpAuthProvider) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	return &oauth2.Transport{
		Source: g.tokenSource,
		Base:   rt,
	}
}

func (g *gcpAuthProvider) Login() error { return nil }

type cachedTokenSource struct {
	source      oauth2.TokenSource
	accessToken string
	expiry      time.Time
	persister   rest.AuthProviderConfigPersister
	cache       map[string]string
}

func newCachedTokenSource(accessToken, expiry string, persister rest.AuthProviderConfigPersister, ts oauth2.TokenSource, cache map[string]string) (*cachedTokenSource, error) {
	var expiryTime time.Time
	if parsedTime, err := time.Parse(time.RFC3339Nano, expiry); err == nil {
		expiryTime = parsedTime
	}
	if cache == nil {
		cache = make(map[string]string)
	}
	return &cachedTokenSource{
		source:      ts,
		accessToken: accessToken,
		expiry:      expiryTime,
		persister:   persister,
		cache:       cache,
	}, nil
}

func (t *cachedTokenSource) Token() (*oauth2.Token, error) {
	tok := &oauth2.Token{
		AccessToken: t.accessToken,
		TokenType:   "Bearer",
		Expiry:      t.expiry,
	}
	if tok.Valid() && !tok.Expiry.IsZero() {
		return tok, nil
	}
	tok, err := t.source.Token()
	if err != nil {
		return nil, err
	}
	if t.persister != nil {
		t.cache["access-token"] = tok.AccessToken
		t.cache["expiry"] = tok.Expiry.Format(time.RFC3339Nano)
		if err := t.persister.Persist(t.cache); err != nil {
			glog.V(4).Infof("Failed to persist token: %v", err)
		}
	}
	return tok, nil
}

type commandTokenSource struct {
	cmd       string
	args      []string
	tokenKey  string
	expiryKey string
	timeFmt   string
}

func newCmdTokenSource(cmd, tokenKey, expiryKey, timeFmt string) (*commandTokenSource, error) {
	if len(timeFmt) == 0 {
		timeFmt = time.RFC3339Nano
	}
	if len(tokenKey) == 0 {
		tokenKey = "{.access_token}"
	}
	if len(expiryKey) == 0 {
		expiryKey = "{.token_expiry}"
	}
	fields := strings.Fields(cmd)
	if len(fields) == 0 {
		return nil, fmt.Errorf("missing access token cmd")
	}
	return &commandTokenSource{
		cmd:       fields[0],
		args:      fields[1:],
		tokenKey:  tokenKey,
		expiryKey: expiryKey,
		timeFmt:   timeFmt,
	}, nil
}

func (c *commandTokenSource) Token() (*oauth2.Token, error) {
	fullCmd := fmt.Sprintf("%s %s", c.cmd, strings.Join(c.args, " "))
	cmd := exec.Command(c.cmd, c.args...)
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("error executing access token command %q: %v", fullCmd, err)
	}
	token, err := c.parseTokenCmdOutput(output)
	if err != nil {
		return nil, fmt.Errorf("error parsing output for access token command %q: %v", fullCmd, err)
	}
	return token, nil
}

func (c *commandTokenSource) parseTokenCmdOutput(output []byte) (*oauth2.Token, error) {
	output, err := yaml.ToJSON(output)
	if err != nil {
		return nil, err
	}
	var data interface{}
	if err := json.Unmarshal(output, &data); err != nil {
		return nil, err
	}

	accessToken, err := parseJSONPath(data, "token-key", c.tokenKey)
	if err != nil {
		return nil, fmt.Errorf("error parsing token-key %q: %v", c.tokenKey, err)
	}
	expiryStr, err := parseJSONPath(data, "expiry-key", c.expiryKey)
	if err != nil {
		return nil, fmt.Errorf("error parsing expiry-key %q: %v", c.expiryKey, err)
	}
	var expiry time.Time
	if t, err := time.Parse(c.timeFmt, expiryStr); err != nil {
		glog.V(4).Infof("Failed to parse token expiry from %s (fmt=%s): %v", expiryStr, c.timeFmt, err)
	} else {
		expiry = t
	}

	return &oauth2.Token{
		AccessToken: accessToken,
		TokenType:   "Bearer",
		Expiry:      expiry,
	}, nil
}

func parseJSONPath(input interface{}, name, template string) (string, error) {
	j := jsonpath.New(name)
	buf := new(bytes.Buffer)
	if err := j.Parse(template); err != nil {
		return "", err
	}
	if err := j.Execute(buf, input); err != nil {
		return "", err
	}
	return buf.String(), nil
}

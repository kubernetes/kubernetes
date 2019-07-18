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
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"sync"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/yaml"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/jsonpath"
	"k8s.io/klog"
)

func init() {
	if err := restclient.RegisterAuthProviderPlugin("gcp", newGCPAuthProvider); err != nil {
		klog.Fatalf("Failed to register gcp auth plugin: %v", err)
	}
}

var (
	// Stubbable for testing
	execCommand = exec.Command

	// defaultScopes:
	// - cloud-platform is the base scope to authenticate to GCP.
	// - userinfo.email is used to authenticate to GKE APIs with gserviceaccount
	//   email instead of numeric uniqueID.
	defaultScopes = []string{
		"https://www.googleapis.com/auth/cloud-platform",
		"https://www.googleapis.com/auth/userinfo.email"}
)

// gcpAuthProvider is an auth provider plugin that uses GCP credentials to provide
// tokens for kubectl to authenticate itself to the apiserver. A sample json config
// is provided below with all recognized options described.
//
// {
//   'auth-provider': {
//     # Required
//     "name": "gcp",
//
//     'config': {
//       # Authentication options
//       # These options are used while getting a token.
//
//       # comma-separated list of GCP API scopes. default value of this field
//       # is "https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email".
// 		 # to override the API scopes, specify this field explicitly.
//       "scopes": "https://www.googleapis.com/auth/cloud-platform"
//
//       # Caching options
//
//       # Raw string data representing cached access token.
//       "access-token": "ya29.CjWdA4GiBPTt",
//       # RFC3339Nano expiration timestamp for cached access token.
//       "expiry": "2016-10-31 22:31:9.123",
//
//       # Command execution options
//       # These options direct the plugin to execute a specified command and parse
//       # token and expiry time from the output of the command.
//
//       # Command to execute for access token. Command output will be parsed as JSON.
//       # If "cmd-args" is not present, this value will be split on whitespace, with
//       # the first element interpreted as the command, remaining elements as args.
//       "cmd-path": "/usr/bin/gcloud",
//
//       # Arguments to pass to command to execute for access token.
//       "cmd-args": "config config-helper --output=json"
//
//       # JSONPath to the string field that represents the access token in
//       # command output. If omitted, defaults to "{.access_token}".
//       "token-key": "{.credential.access_token}",
//
//       # JSONPath to the string field that represents expiration timestamp
//       # of the access token in the command output. If omitted, defaults to
//       # "{.token_expiry}"
//       "expiry-key": ""{.credential.token_expiry}",
//
//       # golang reference time in the format that the expiration timestamp uses.
//       # If omitted, defaults to time.RFC3339Nano
//       "time-fmt": "2006-01-02 15:04:05.999999999"
//     }
//   }
// }
//
type gcpAuthProvider struct {
	tokenSource oauth2.TokenSource
	persister   restclient.AuthProviderConfigPersister
}

func newGCPAuthProvider(_ string, gcpConfig map[string]string, persister restclient.AuthProviderConfigPersister) (restclient.AuthProvider, error) {
	tknType, err := tokenSourceType(gcpConfig)
	if err != nil {
		return nil, err
	}
	ts, err := tokenSource(tknType, gcpConfig)
	if err != nil {
		return nil, err
	}
	cts, err := newCachedTokenSource(gcpConfig["access-token"], gcpConfig["expiry"], persister, ts, gcpConfig)
	if err != nil {
		return nil, err
	}
	return &gcpAuthProvider{cts, persister}, nil
}

func tokenSourceType(gcpConfig map[string]string) (string, error) {
	var types []string
	if _, ok := gcpConfig[TokenFromCmd]; ok {
		types = append(types, TokenFromCmd)
	}
	if _, ok := gcpConfig[TokenFromString]; ok {
		types = append(types, TokenFromString)
	}
	if _, ok := gcpConfig[TokenFromEnvPath]; ok {
		types = append(types, TokenFromEnvPath)
	}
	if _, ok := gcpConfig[TokenFromEnvString]; ok {
		types = append(types, TokenFromEnvString)
	}
	if len(types) > 1 {
		return "", errors.New("more than a single auth token types provided")
	}
	if len(types) == 1 {
		return types[0], nil
	}
	return "", nil
}

const (
	// TokenFromCmd runs a command to get the token.
	TokenFromCmd = "cmd-path"
	// TokenFromString contains the token string.
	TokenFromString = "string"
	// TokenFromPath points to a file containing the token.
	TokenFromPath = "path"
	// TokenFromEnvPath is an env which points to a file that contains the token.
	TokenFromEnvPath = "env-path"
	// TokenFromEnvString is an env which contains the token string.
	TokenFromEnvString = "env-string"
)

func tokenSource(tknType string, gcpConfig map[string]string) (oauth2.TokenSource, error) {
	tknSrc := gcpConfig[tknType]
	if tknType != "" && tknSrc == "" {
		return nil, fmt.Errorf("missing access token string")
	}
	scopes := parseScopes(gcpConfig)

	switch tknType {
	case TokenFromCmd:
		if gcpConfig["scopes"] != "" {
			return nil, fmt.Errorf("scopes can only be used when kubectl is using a gcp service account key")
		}
		var args []string
		if cmdArgs, ok := gcpConfig["cmd-args"]; ok {
			args = strings.Fields(cmdArgs)
		} else {
			fields := strings.Fields(tknSrc)
			tknSrc = fields[0]
			args = fields[1:]
		}
		return newCmdTokenSource(tknSrc, args, gcpConfig["token-key"], gcpConfig["expiry-key"], gcpConfig["time-fmt"]), nil
	case TokenFromString:
		return newStrTokenSource(tknSrc, scopes)
	case TokenFromPath:
		return newPathTokenSource(tknSrc, scopes)
	case TokenFromEnvPath:
		return newENVPathTokenSource(tknSrc, scopes)
	case TokenFromEnvString:
		return newENVStringTokenSource(tknSrc, scopes)
	}

	// When everything else failed try to use the default
	// Google Application Credentials-based token source.
	ts, err := google.DefaultTokenSource(context.Background(), scopes...)
	if err != nil {
		return nil, fmt.Errorf("cannot construct google default token source: %v", err)
	}
	return ts, nil
}

// parseScopes constructs a list of scopes that should be included in token source
// from the config map.
func parseScopes(gcpConfig map[string]string) []string {
	scopes, ok := gcpConfig["scopes"]
	if !ok {
		return defaultScopes
	}
	if scopes == "" {
		return []string{}
	}
	return strings.Split(gcpConfig["scopes"], ",")
}

func (g *gcpAuthProvider) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	var resetCache map[string]string
	if cts, ok := g.tokenSource.(*cachedTokenSource); ok {
		resetCache = cts.baseCache()
	} else {
		resetCache = make(map[string]string)
	}
	return &conditionalTransport{&oauth2.Transport{Source: g.tokenSource, Base: rt}, g.persister, resetCache}
}

func (g *gcpAuthProvider) Login() error { return nil }

type cachedTokenSource struct {
	lk          sync.Mutex
	source      oauth2.TokenSource
	accessToken string
	expiry      time.Time
	persister   restclient.AuthProviderConfigPersister
	cache       map[string]string
}

func newCachedTokenSource(accessToken, expiry string, persister restclient.AuthProviderConfigPersister, ts oauth2.TokenSource, cache map[string]string) (*cachedTokenSource, error) {
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
	tok := t.cachedToken()
	if tok.Valid() && !tok.Expiry.IsZero() {
		return tok, nil
	}
	tok, err := t.source.Token()
	if err != nil {
		return nil, err
	}
	cache := t.update(tok)
	if t.persister != nil {
		if err := t.persister.Persist(cache); err != nil {
			klog.V(4).Infof("Failed to persist token: %v", err)
		}
	}
	return tok, nil
}

func (t *cachedTokenSource) cachedToken() *oauth2.Token {
	t.lk.Lock()
	defer t.lk.Unlock()
	return &oauth2.Token{
		AccessToken: t.accessToken,
		TokenType:   "Bearer",
		Expiry:      t.expiry,
	}
}

func (t *cachedTokenSource) update(tok *oauth2.Token) map[string]string {
	t.lk.Lock()
	defer t.lk.Unlock()
	t.accessToken = tok.AccessToken
	t.expiry = tok.Expiry
	ret := map[string]string{}
	for k, v := range t.cache {
		ret[k] = v
	}
	ret["access-token"] = t.accessToken
	ret["expiry"] = t.expiry.Format(time.RFC3339Nano)
	return ret
}

// baseCache is the base configuration value for this TokenSource, without any cached ephemeral tokens.
func (t *cachedTokenSource) baseCache() map[string]string {
	t.lk.Lock()
	defer t.lk.Unlock()
	ret := map[string]string{}
	for k, v := range t.cache {
		ret[k] = v
	}
	delete(ret, "access-token")
	delete(ret, "expiry")
	return ret
}

type stringTokenSource struct {
	tk oauth2.TokenSource
}

func (c *stringTokenSource) Token() (*oauth2.Token, error) {
	return c.tk.Token()
}

func newStrTokenSource(tkn string, scopes []string) (*stringTokenSource, error) {
	tkn, err := decodeBase64String(tkn)
	if err != nil {
		return nil, fmt.Errorf("decoding the token string: %v", err)

	}
	ts, err := google.CredentialsFromJSON(context.Background(), []byte(tkn), scopes...)
	if err != nil {
		return nil, fmt.Errorf("cannot construct google default token source: %v", err)
	}
	return &stringTokenSource{tk: ts.TokenSource}, nil
}

func newPathTokenSource(tkn string, scopes []string) (*stringTokenSource, error) {
	content, err := ioutil.ReadFile(tkn)
	if err != nil {
		return nil, fmt.Errorf("reading the token file: %v", err)
	}
	return newStrTokenSource(string(content), scopes)
}

func newENVPathTokenSource(tkn string, scopes []string) (*stringTokenSource, error) {
	tkn = os.Getenv(tkn)
	return newPathTokenSource(tkn, scopes)
}

func newENVStringTokenSource(tkn string, scopes []string) (*stringTokenSource, error) {
	tkn = os.Getenv(tkn)
	return newStrTokenSource(tkn, scopes)
}

type commandTokenSource struct {
	cmd       string
	args      []string
	tokenKey  string
	expiryKey string
	timeFmt   string
}

func newCmdTokenSource(cmd string, args []string, tokenKey, expiryKey, timeFmt string) *commandTokenSource {
	if len(timeFmt) == 0 {
		timeFmt = time.RFC3339Nano
	}
	if len(tokenKey) == 0 {
		tokenKey = "{.access_token}"
	}
	if len(expiryKey) == 0 {
		expiryKey = "{.token_expiry}"
	}
	return &commandTokenSource{
		cmd:       cmd,
		args:      args,
		tokenKey:  tokenKey,
		expiryKey: expiryKey,
		timeFmt:   timeFmt,
	}
}

func (c *commandTokenSource) Token() (*oauth2.Token, error) {
	fullCmd := strings.Join(append([]string{c.cmd}, c.args...), " ")
	cmd := execCommand(c.cmd, c.args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	output, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("error executing access token command %q: err=%v output=%s stderr=%s", fullCmd, err, output, string(stderr.Bytes()))
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
		return nil, fmt.Errorf("error parsing token-key %q from %q: %v", c.tokenKey, string(output), err)
	}
	expiryStr, err := parseJSONPath(data, "expiry-key", c.expiryKey)
	if err != nil {
		return nil, fmt.Errorf("error parsing expiry-key %q from %q: %v", c.expiryKey, string(output), err)
	}
	var expiry time.Time
	if t, err := time.Parse(c.timeFmt, expiryStr); err != nil {
		klog.V(4).Infof("Failed to parse token expiry from %s (fmt=%s): %v", expiryStr, c.timeFmt, err)
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

// decodeBase64String checks if a stirng is base64 encoded and
// returns the decoded value.
func decodeBase64String(str string) (string, error) {
	// Check is string is base64 encoded.
	encoded, err := regexp.MatchString("^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)?$", str)
	if err != nil {
		return "", err
	}
	if encoded {
		t, err := base64.StdEncoding.DecodeString(str)
		if err != nil {
			return "nil", err
		}
		str = string(t)
	}
	return str, nil
}

type conditionalTransport struct {
	oauthTransport *oauth2.Transport
	persister      restclient.AuthProviderConfigPersister
	resetCache     map[string]string
}

var _ net.RoundTripperWrapper = &conditionalTransport{}

func (t *conditionalTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("Authorization")) != 0 {
		return t.oauthTransport.Base.RoundTrip(req)
	}

	res, err := t.oauthTransport.RoundTrip(req)

	if err != nil {
		return nil, err
	}

	if res.StatusCode == 401 {
		klog.V(4).Infof("The credentials that were supplied are invalid for the target cluster")
		t.persister.Persist(t.resetCache)
	}

	return res, nil
}

func (t *conditionalTransport) WrappedRoundTripper() http.RoundTripper { return t.oauthTransport.Base }

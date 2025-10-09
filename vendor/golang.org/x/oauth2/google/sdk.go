// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/user"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"golang.org/x/oauth2"
)

type sdkCredentials struct {
	Data []struct {
		Credential struct {
			ClientID     string     `json:"client_id"`
			ClientSecret string     `json:"client_secret"`
			AccessToken  string     `json:"access_token"`
			RefreshToken string     `json:"refresh_token"`
			TokenExpiry  *time.Time `json:"token_expiry"`
		} `json:"credential"`
		Key struct {
			Account string `json:"account"`
			Scope   string `json:"scope"`
		} `json:"key"`
	}
}

// An SDKConfig provides access to tokens from an account already
// authorized via the Google Cloud SDK.
type SDKConfig struct {
	conf         oauth2.Config
	initialToken *oauth2.Token
}

// NewSDKConfig creates an SDKConfig for the given Google Cloud SDK
// account. If account is empty, the account currently active in
// Google Cloud SDK properties is used.
// Google Cloud SDK credentials must be created by running `gcloud auth`
// before using this function.
// The Google Cloud SDK is available at https://cloud.google.com/sdk/.
func NewSDKConfig(account string) (*SDKConfig, error) {
	configPath, err := sdkConfigPath()
	if err != nil {
		return nil, fmt.Errorf("oauth2/google: error getting SDK config path: %v", err)
	}
	credentialsPath := filepath.Join(configPath, "credentials")
	f, err := os.Open(credentialsPath)
	if err != nil {
		return nil, fmt.Errorf("oauth2/google: failed to load SDK credentials: %v", err)
	}
	defer f.Close()

	var c sdkCredentials
	if err := json.NewDecoder(f).Decode(&c); err != nil {
		return nil, fmt.Errorf("oauth2/google: failed to decode SDK credentials from %q: %v", credentialsPath, err)
	}
	if len(c.Data) == 0 {
		return nil, fmt.Errorf("oauth2/google: no credentials found in %q, run `gcloud auth login` to create one", credentialsPath)
	}
	if account == "" {
		propertiesPath := filepath.Join(configPath, "properties")
		f, err := os.Open(propertiesPath)
		if err != nil {
			return nil, fmt.Errorf("oauth2/google: failed to load SDK properties: %v", err)
		}
		defer f.Close()
		ini, err := parseINI(f)
		if err != nil {
			return nil, fmt.Errorf("oauth2/google: failed to parse SDK properties %q: %v", propertiesPath, err)
		}
		core, ok := ini["core"]
		if !ok {
			return nil, fmt.Errorf("oauth2/google: failed to find [core] section in %v", ini)
		}
		active, ok := core["account"]
		if !ok {
			return nil, fmt.Errorf("oauth2/google: failed to find %q attribute in %v", "account", core)
		}
		account = active
	}

	for _, d := range c.Data {
		if account == "" || d.Key.Account == account {
			if d.Credential.AccessToken == "" && d.Credential.RefreshToken == "" {
				return nil, fmt.Errorf("oauth2/google: no token available for account %q", account)
			}
			var expiry time.Time
			if d.Credential.TokenExpiry != nil {
				expiry = *d.Credential.TokenExpiry
			}
			return &SDKConfig{
				conf: oauth2.Config{
					ClientID:     d.Credential.ClientID,
					ClientSecret: d.Credential.ClientSecret,
					Scopes:       strings.Split(d.Key.Scope, " "),
					Endpoint:     Endpoint,
					RedirectURL:  "oob",
				},
				initialToken: &oauth2.Token{
					AccessToken:  d.Credential.AccessToken,
					RefreshToken: d.Credential.RefreshToken,
					Expiry:       expiry,
				},
			}, nil
		}
	}
	return nil, fmt.Errorf("oauth2/google: no such credentials for account %q", account)
}

// Client returns an HTTP client using Google Cloud SDK credentials to
// authorize requests. The token will auto-refresh as necessary. The
// underlying http.RoundTripper will be obtained using the provided
// context. The returned client and its Transport should not be
// modified.
func (c *SDKConfig) Client(ctx context.Context) *http.Client {
	return &http.Client{
		Transport: &oauth2.Transport{
			Source: c.TokenSource(ctx),
		},
	}
}

// TokenSource returns an oauth2.TokenSource that retrieve tokens from
// Google Cloud SDK credentials using the provided context.
// It will returns the current access token stored in the credentials,
// and refresh it when it expires, but it won't update the credentials
// with the new access token.
func (c *SDKConfig) TokenSource(ctx context.Context) oauth2.TokenSource {
	return c.conf.TokenSource(ctx, c.initialToken)
}

// Scopes are the OAuth 2.0 scopes the current account is authorized for.
func (c *SDKConfig) Scopes() []string {
	return c.conf.Scopes
}

func parseINI(ini io.Reader) (map[string]map[string]string, error) {
	result := map[string]map[string]string{
		"": {}, // root section
	}
	scanner := bufio.NewScanner(ini)
	currentSection := ""
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if strings.HasPrefix(line, ";") {
			// comment.
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			currentSection = strings.TrimSpace(line[1 : len(line)-1])
			result[currentSection] = map[string]string{}
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 && parts[0] != "" {
			result[currentSection][strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error scanning ini: %v", err)
	}
	return result, nil
}

// sdkConfigPath tries to guess where the gcloud config is located.
// It can be overridden during tests.
var sdkConfigPath = func() (string, error) {
	if runtime.GOOS == "windows" {
		return filepath.Join(os.Getenv("APPDATA"), "gcloud"), nil
	}
	homeDir := guessUnixHomeDir()
	if homeDir == "" {
		return "", errors.New("unable to get current user home directory: os/user lookup failed; $HOME is empty")
	}
	return filepath.Join(homeDir, ".config", "gcloud"), nil
}

func guessUnixHomeDir() string {
	// Prefer $HOME over user.Current due to glibc bug: golang.org/issue/13470
	if v := os.Getenv("HOME"); v != "" {
		return v
	}
	// Else, fall back to user.Current:
	if u, err := user.Current(); err == nil {
		return u.HomeDir
	}
	return ""
}

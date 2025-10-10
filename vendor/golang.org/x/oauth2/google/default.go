// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package google

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"time"

	"cloud.google.com/go/compute/metadata"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/authhandler"
)

const (
	adcSetupURL           = "https://cloud.google.com/docs/authentication/external/set-up-adc"
	defaultUniverseDomain = "googleapis.com"
)

// Credentials holds Google credentials, including "Application Default Credentials".
// For more details, see:
// https://developers.google.com/accounts/docs/application-default-credentials
// Credentials from external accounts (workload identity federation) are used to
// identify a particular application from an on-prem or non-Google Cloud platform
// including Amazon Web Services (AWS), Microsoft Azure or any identity provider
// that supports OpenID Connect (OIDC).
type Credentials struct {
	ProjectID   string // may be empty
	TokenSource oauth2.TokenSource

	// JSON contains the raw bytes from a JSON credentials file.
	// This field may be nil if authentication is provided by the
	// environment and not with a credentials file, e.g. when code is
	// running on Google Cloud Platform.
	JSON []byte

	// UniverseDomainProvider returns the default service domain for a given
	// Cloud universe. Optional.
	//
	// On GCE, UniverseDomainProvider should return the universe domain value
	// from Google Compute Engine (GCE)'s metadata server. See also [The attached service
	// account](https://cloud.google.com/docs/authentication/application-default-credentials#attached-sa).
	// If the GCE metadata server returns a 404 error, the default universe
	// domain value should be returned. If the GCE metadata server returns an
	// error other than 404, the error should be returned.
	UniverseDomainProvider func() (string, error)

	udMu sync.Mutex // guards universeDomain
	// universeDomain is the default service domain for a given Cloud universe.
	universeDomain string
}

// UniverseDomain returns the default service domain for a given Cloud universe.
//
// The default value is "googleapis.com".
//
// Deprecated: Use instead (*Credentials).GetUniverseDomain(), which supports
// obtaining the universe domain when authenticating via the GCE metadata server.
// Unlike GetUniverseDomain, this method, UniverseDomain, will always return the
// default value when authenticating via the GCE metadata server.
// See also [The attached service account](https://cloud.google.com/docs/authentication/application-default-credentials#attached-sa).
func (c *Credentials) UniverseDomain() string {
	if c.universeDomain == "" {
		return defaultUniverseDomain
	}
	return c.universeDomain
}

// GetUniverseDomain returns the default service domain for a given Cloud
// universe. If present, UniverseDomainProvider will be invoked and its return
// value will be cached.
//
// The default value is "googleapis.com".
func (c *Credentials) GetUniverseDomain() (string, error) {
	c.udMu.Lock()
	defer c.udMu.Unlock()
	if c.universeDomain == "" && c.UniverseDomainProvider != nil {
		// On Google Compute Engine, an App Engine standard second generation
		// runtime, or App Engine flexible, use an externally provided function
		// to request the universe domain from the metadata server.
		ud, err := c.UniverseDomainProvider()
		if err != nil {
			return "", err
		}
		c.universeDomain = ud
	}
	// If no UniverseDomainProvider (meaning not on Google Compute Engine), or
	// in case of any (non-error) empty return value from
	// UniverseDomainProvider, set the default universe domain.
	if c.universeDomain == "" {
		c.universeDomain = defaultUniverseDomain
	}
	return c.universeDomain, nil
}

// DefaultCredentials is the old name of Credentials.
//
// Deprecated: use Credentials instead.
type DefaultCredentials = Credentials

// CredentialsParams holds user supplied parameters that are used together
// with a credentials file for building a Credentials object.
type CredentialsParams struct {
	// Scopes is the list OAuth scopes. Required.
	// Example: https://www.googleapis.com/auth/cloud-platform
	Scopes []string

	// Subject is the user email used for domain wide delegation (see
	// https://developers.google.com/identity/protocols/oauth2/service-account#delegatingauthority).
	// Optional.
	Subject string

	// AuthHandler is the AuthorizationHandler used for 3-legged OAuth flow. Required for 3LO flow.
	AuthHandler authhandler.AuthorizationHandler

	// State is a unique string used with AuthHandler. Required for 3LO flow.
	State string

	// PKCE is used to support PKCE flow. Optional for 3LO flow.
	PKCE *authhandler.PKCEParams

	// The OAuth2 TokenURL default override. This value overrides the default TokenURL,
	// unless explicitly specified by the credentials config file. Optional.
	TokenURL string

	// EarlyTokenRefresh is the amount of time before a token expires that a new
	// token will be preemptively fetched. If unset the default value is 10
	// seconds.
	//
	// Note: This option is currently only respected when using credentials
	// fetched from the GCE metadata server.
	EarlyTokenRefresh time.Duration

	// UniverseDomain is the default service domain for a given Cloud universe.
	// Only supported in authentication flows that support universe domains.
	// This value takes precedence over a universe domain explicitly specified
	// in a credentials config file or by the GCE metadata server. Optional.
	UniverseDomain string
}

func (params CredentialsParams) deepCopy() CredentialsParams {
	paramsCopy := params
	paramsCopy.Scopes = make([]string, len(params.Scopes))
	copy(paramsCopy.Scopes, params.Scopes)
	return paramsCopy
}

// DefaultClient returns an HTTP Client that uses the
// DefaultTokenSource to obtain authentication credentials.
func DefaultClient(ctx context.Context, scope ...string) (*http.Client, error) {
	ts, err := DefaultTokenSource(ctx, scope...)
	if err != nil {
		return nil, err
	}
	return oauth2.NewClient(ctx, ts), nil
}

// DefaultTokenSource returns the token source for
// "Application Default Credentials".
// It is a shortcut for FindDefaultCredentials(ctx, scope).TokenSource.
func DefaultTokenSource(ctx context.Context, scope ...string) (oauth2.TokenSource, error) {
	creds, err := FindDefaultCredentials(ctx, scope...)
	if err != nil {
		return nil, err
	}
	return creds.TokenSource, nil
}

// FindDefaultCredentialsWithParams searches for "Application Default Credentials".
//
// It looks for credentials in the following places,
// preferring the first location found:
//
//  1. A JSON file whose path is specified by the
//     GOOGLE_APPLICATION_CREDENTIALS environment variable.
//     For workload identity federation, refer to
//     https://cloud.google.com/iam/docs/how-to#using-workload-identity-federation on
//     how to generate the JSON configuration file for on-prem/non-Google cloud
//     platforms.
//  2. A JSON file in a location known to the gcloud command-line tool.
//     On Windows, this is %APPDATA%/gcloud/application_default_credentials.json.
//     On other systems, $HOME/.config/gcloud/application_default_credentials.json.
//  3. On Google Compute Engine, Google App Engine standard second generation runtimes
//     (>= Go 1.11), and Google App Engine flexible environment, it fetches
//     credentials from the metadata server.
func FindDefaultCredentialsWithParams(ctx context.Context, params CredentialsParams) (*Credentials, error) {
	// Make defensive copy of the slices in params.
	params = params.deepCopy()

	// First, try the environment variable.
	const envVar = "GOOGLE_APPLICATION_CREDENTIALS"
	if filename := os.Getenv(envVar); filename != "" {
		creds, err := readCredentialsFile(ctx, filename, params)
		if err != nil {
			return nil, fmt.Errorf("google: error getting credentials using %v environment variable: %v", envVar, err)
		}
		return creds, nil
	}

	// Second, try a well-known file.
	filename := wellKnownFile()
	if b, err := os.ReadFile(filename); err == nil {
		return CredentialsFromJSONWithParams(ctx, b, params)
	}

	// Third, if we're on Google Compute Engine, an App Engine standard second generation runtime,
	// or App Engine flexible, use the metadata server.
	if metadata.OnGCE() {
		id, _ := metadata.ProjectID()
		universeDomainProvider := func() (string, error) {
			universeDomain, err := metadata.Get("universe/universe_domain")
			if err != nil {
				if _, ok := err.(metadata.NotDefinedError); ok {
					// http.StatusNotFound (404)
					return defaultUniverseDomain, nil
				} else {
					return "", err
				}
			}
			return universeDomain, nil
		}
		return &Credentials{
			ProjectID:              id,
			TokenSource:            computeTokenSource("", params.EarlyTokenRefresh, params.Scopes...),
			UniverseDomainProvider: universeDomainProvider,
			universeDomain:         params.UniverseDomain,
		}, nil
	}

	// None are found; return helpful error.
	return nil, fmt.Errorf("google: could not find default credentials. See %v for more information", adcSetupURL)
}

// FindDefaultCredentials invokes FindDefaultCredentialsWithParams with the specified scopes.
func FindDefaultCredentials(ctx context.Context, scopes ...string) (*Credentials, error) {
	var params CredentialsParams
	params.Scopes = scopes
	return FindDefaultCredentialsWithParams(ctx, params)
}

// CredentialsFromJSONWithParams obtains Google credentials from a JSON value. The JSON can
// represent either a Google Developers Console client_credentials.json file (as in ConfigFromJSON),
// a Google Developers service account key file, a gcloud user credentials file (a.k.a. refresh
// token JSON), or the JSON configuration file for workload identity federation in non-Google cloud
// platforms (see https://cloud.google.com/iam/docs/how-to#using-workload-identity-federation).
//
// Important: If you accept a credential configuration (credential JSON/File/Stream) from an
// external source for authentication to Google Cloud Platform, you must validate it before
// providing it to any Google API or library. Providing an unvalidated credential configuration to
// Google APIs can compromise the security of your systems and data. For more information, refer to
// [Validate credential configurations from external sources](https://cloud.google.com/docs/authentication/external/externally-sourced-credentials).
func CredentialsFromJSONWithParams(ctx context.Context, jsonData []byte, params CredentialsParams) (*Credentials, error) {
	// Make defensive copy of the slices in params.
	params = params.deepCopy()

	// First, attempt to parse jsonData as a Google Developers Console client_credentials.json.
	config, _ := ConfigFromJSON(jsonData, params.Scopes...)
	if config != nil {
		return &Credentials{
			ProjectID:   "",
			TokenSource: authhandler.TokenSourceWithPKCE(ctx, config, params.State, params.AuthHandler, params.PKCE),
			JSON:        jsonData,
		}, nil
	}

	// Otherwise, parse jsonData as one of the other supported credentials files.
	var f credentialsFile
	if err := json.Unmarshal(jsonData, &f); err != nil {
		return nil, err
	}

	universeDomain := f.UniverseDomain
	if params.UniverseDomain != "" {
		universeDomain = params.UniverseDomain
	}
	// Authorized user credentials are only supported in the googleapis.com universe.
	if f.Type == userCredentialsKey {
		universeDomain = defaultUniverseDomain
	}

	ts, err := f.tokenSource(ctx, params)
	if err != nil {
		return nil, err
	}
	ts = newErrWrappingTokenSource(ts)
	return &Credentials{
		ProjectID:      f.ProjectID,
		TokenSource:    ts,
		JSON:           jsonData,
		universeDomain: universeDomain,
	}, nil
}

// CredentialsFromJSON invokes CredentialsFromJSONWithParams with the specified scopes.
//
// Important: If you accept a credential configuration (credential JSON/File/Stream) from an
// external source for authentication to Google Cloud Platform, you must validate it before
// providing it to any Google API or library. Providing an unvalidated credential configuration to
// Google APIs can compromise the security of your systems and data. For more information, refer to
// [Validate credential configurations from external sources](https://cloud.google.com/docs/authentication/external/externally-sourced-credentials).
func CredentialsFromJSON(ctx context.Context, jsonData []byte, scopes ...string) (*Credentials, error) {
	var params CredentialsParams
	params.Scopes = scopes
	return CredentialsFromJSONWithParams(ctx, jsonData, params)
}

func wellKnownFile() string {
	const f = "application_default_credentials.json"
	if runtime.GOOS == "windows" {
		return filepath.Join(os.Getenv("APPDATA"), "gcloud", f)
	}
	return filepath.Join(guessUnixHomeDir(), ".config", "gcloud", f)
}

func readCredentialsFile(ctx context.Context, filename string, params CredentialsParams) (*Credentials, error) {
	b, err := os.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return CredentialsFromJSONWithParams(ctx, b, params)
}

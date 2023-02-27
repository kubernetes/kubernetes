//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package azidentity

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strings"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
)

const credNameAzureCLI = "AzureCLICredential"

// used by tests to fake invoking the CLI
type azureCLITokenProvider func(ctx context.Context, resource string, tenantID string) ([]byte, error)

// AzureCLICredentialOptions contains optional parameters for AzureCLICredential.
type AzureCLICredentialOptions struct {
	// TenantID identifies the tenant the credential should authenticate in.
	// Defaults to the CLI's default tenant, which is typically the home tenant of the logged in user.
	TenantID string

	tokenProvider azureCLITokenProvider
}

// init returns an instance of AzureCLICredentialOptions initialized with default values.
func (o *AzureCLICredentialOptions) init() {
	if o.tokenProvider == nil {
		o.tokenProvider = defaultTokenProvider()
	}
}

// AzureCLICredential authenticates as the identity logged in to the Azure CLI.
type AzureCLICredential struct {
	tokenProvider azureCLITokenProvider
	tenantID      string
}

// NewAzureCLICredential constructs an AzureCLICredential. Pass nil to accept default options.
func NewAzureCLICredential(options *AzureCLICredentialOptions) (*AzureCLICredential, error) {
	cp := AzureCLICredentialOptions{}
	if options != nil {
		cp = *options
	}
	cp.init()
	return &AzureCLICredential{
		tokenProvider: cp.tokenProvider,
		tenantID:      cp.TenantID,
	}, nil
}

// GetToken requests a token from the Azure CLI. This credential doesn't cache tokens, so every call invokes the CLI.
// This method is called automatically by Azure SDK clients.
func (c *AzureCLICredential) GetToken(ctx context.Context, opts policy.TokenRequestOptions) (azcore.AccessToken, error) {
	if len(opts.Scopes) != 1 {
		return azcore.AccessToken{}, errors.New(credNameAzureCLI + ": GetToken() requires exactly one scope")
	}
	// CLI expects an AAD v1 resource, not a v2 scope
	scope := strings.TrimSuffix(opts.Scopes[0], defaultSuffix)
	at, err := c.authenticate(ctx, scope)
	if err != nil {
		return azcore.AccessToken{}, err
	}
	logGetTokenSuccess(c, opts)
	return at, nil
}

const timeoutCLIRequest = 10 * time.Second

func (c *AzureCLICredential) authenticate(ctx context.Context, resource string) (azcore.AccessToken, error) {
	output, err := c.tokenProvider(ctx, resource, c.tenantID)
	if err != nil {
		return azcore.AccessToken{}, err
	}

	return c.createAccessToken(output)
}

func defaultTokenProvider() func(ctx context.Context, resource string, tenantID string) ([]byte, error) {
	return func(ctx context.Context, resource string, tenantID string) ([]byte, error) {
		match, err := regexp.MatchString("^[0-9a-zA-Z-.:/]+$", resource)
		if err != nil {
			return nil, err
		}
		if !match {
			return nil, fmt.Errorf(`%s: unexpected scope "%s". Only alphanumeric characters and ".", ";", "-", and "/" are allowed`, credNameAzureCLI, resource)
		}

		ctx, cancel := context.WithTimeout(ctx, timeoutCLIRequest)
		defer cancel()

		commandLine := "az account get-access-token -o json --resource " + resource
		if tenantID != "" {
			commandLine += " --tenant " + tenantID
		}
		var cliCmd *exec.Cmd
		if runtime.GOOS == "windows" {
			dir := os.Getenv("SYSTEMROOT")
			if dir == "" {
				return nil, newCredentialUnavailableError(credNameAzureCLI, "environment variable 'SYSTEMROOT' has no value")
			}
			cliCmd = exec.CommandContext(ctx, "cmd.exe", "/c", commandLine)
			cliCmd.Dir = dir
		} else {
			cliCmd = exec.CommandContext(ctx, "/bin/sh", "-c", commandLine)
			cliCmd.Dir = "/bin"
		}
		cliCmd.Env = os.Environ()
		var stderr bytes.Buffer
		cliCmd.Stderr = &stderr

		output, err := cliCmd.Output()
		if err != nil {
			msg := stderr.String()
			var exErr *exec.ExitError
			if errors.As(err, &exErr) && exErr.ExitCode() == 127 || strings.HasPrefix(msg, "'az' is not recognized") {
				msg = "Azure CLI not found on path"
			}
			if msg == "" {
				msg = err.Error()
			}
			return nil, newCredentialUnavailableError(credNameAzureCLI, msg)
		}

		return output, nil
	}
}

func (c *AzureCLICredential) createAccessToken(tk []byte) (azcore.AccessToken, error) {
	t := struct {
		AccessToken      string `json:"accessToken"`
		Authority        string `json:"_authority"`
		ClientID         string `json:"_clientId"`
		ExpiresOn        string `json:"expiresOn"`
		IdentityProvider string `json:"identityProvider"`
		IsMRRT           bool   `json:"isMRRT"`
		RefreshToken     string `json:"refreshToken"`
		Resource         string `json:"resource"`
		TokenType        string `json:"tokenType"`
		UserID           string `json:"userId"`
	}{}
	err := json.Unmarshal(tk, &t)
	if err != nil {
		return azcore.AccessToken{}, err
	}

	tokenExpirationDate, err := parseExpirationDate(t.ExpiresOn)
	if err != nil {
		return azcore.AccessToken{}, fmt.Errorf("Error parsing Token Expiration Date %q: %+v", t.ExpiresOn, err)
	}

	converted := azcore.AccessToken{
		Token:     t.AccessToken,
		ExpiresOn: *tokenExpirationDate,
	}
	return converted, nil
}

// parseExpirationDate parses either a Azure CLI or CloudShell date into a time object
func parseExpirationDate(input string) (*time.Time, error) {
	// CloudShell (and potentially the Azure CLI in future)
	expirationDate, cloudShellErr := time.Parse(time.RFC3339, input)
	if cloudShellErr != nil {
		// Azure CLI (Python) e.g. 2017-08-31 19:48:57.998857 (plus the local timezone)
		const cliFormat = "2006-01-02 15:04:05.999999"
		expirationDate, cliErr := time.ParseInLocation(cliFormat, input, time.Local)
		if cliErr != nil {
			return nil, fmt.Errorf("Error parsing expiration date %q.\n\nCloudShell Error: \n%+v\n\nCLI Error:\n%+v", input, cloudShellErr, cliErr)
		}
		return &expirationDate, nil
	}
	return &expirationDate, nil
}

var _ azcore.TokenCredential = (*AzureCLICredential)(nil)

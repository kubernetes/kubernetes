//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package azidentity

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Azure/azure-sdk-for-go/sdk/azcore"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/policy"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/runtime"
	"github.com/Azure/azure-sdk-for-go/sdk/azcore/streaming"
	"github.com/Azure/azure-sdk-for-go/sdk/internal/log"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/confidential"
)

const (
	arcIMDSEndpoint          = "IMDS_ENDPOINT"
	identityEndpoint         = "IDENTITY_ENDPOINT"
	identityHeader           = "IDENTITY_HEADER"
	identityServerThumbprint = "IDENTITY_SERVER_THUMBPRINT"
	headerMetadata           = "Metadata"
	imdsEndpoint             = "http://169.254.169.254/metadata/identity/oauth2/token"
	msiEndpoint              = "MSI_ENDPOINT"
	imdsAPIVersion           = "2018-02-01"
	azureArcAPIVersion       = "2019-08-15"
	serviceFabricAPIVersion  = "2019-07-01-preview"

	qpClientID = "client_id"
	qpResID    = "mi_res_id"
)

type msiType int

const (
	msiTypeAppService msiType = iota
	msiTypeAzureArc
	msiTypeCloudShell
	msiTypeIMDS
	msiTypeServiceFabric
)

// managedIdentityClient provides the base for authenticating in managed identity environments
// This type includes an runtime.Pipeline and TokenCredentialOptions.
type managedIdentityClient struct {
	pipeline    runtime.Pipeline
	msiType     msiType
	endpoint    string
	id          ManagedIDKind
	imdsTimeout time.Duration
}

type wrappedNumber json.Number

func (n *wrappedNumber) UnmarshalJSON(b []byte) error {
	c := string(b)
	if c == "\"\"" {
		return nil
	}
	return json.Unmarshal(b, (*json.Number)(n))
}

// setIMDSRetryOptionDefaults sets zero-valued fields to default values appropriate for IMDS
func setIMDSRetryOptionDefaults(o *policy.RetryOptions) {
	if o.MaxRetries == 0 {
		o.MaxRetries = 5
	}
	if o.MaxRetryDelay == 0 {
		o.MaxRetryDelay = 1 * time.Minute
	}
	if o.RetryDelay == 0 {
		o.RetryDelay = 2 * time.Second
	}
	if o.StatusCodes == nil {
		o.StatusCodes = []int{
			// IMDS docs recommend retrying 404, 429 and all 5xx
			// https://docs.microsoft.com/azure/active-directory/managed-identities-azure-resources/how-to-use-vm-token#error-handling
			http.StatusNotFound,                      // 404
			http.StatusTooManyRequests,               // 429
			http.StatusInternalServerError,           // 500
			http.StatusNotImplemented,                // 501
			http.StatusBadGateway,                    // 502
			http.StatusGatewayTimeout,                // 504
			http.StatusHTTPVersionNotSupported,       // 505
			http.StatusVariantAlsoNegotiates,         // 506
			http.StatusInsufficientStorage,           // 507
			http.StatusLoopDetected,                  // 508
			http.StatusNotExtended,                   // 510
			http.StatusNetworkAuthenticationRequired, // 511
		}
	}
	if o.TryTimeout == 0 {
		o.TryTimeout = 1 * time.Minute
	}
}

// newManagedIdentityClient creates a new instance of the ManagedIdentityClient with the ManagedIdentityCredentialOptions
// that are passed into it along with a default pipeline.
// options: ManagedIdentityCredentialOptions configure policies for the pipeline and the authority host that
// will be used to retrieve tokens and authenticate
func newManagedIdentityClient(options *ManagedIdentityCredentialOptions) (*managedIdentityClient, error) {
	if options == nil {
		options = &ManagedIdentityCredentialOptions{}
	}
	cp := options.ClientOptions
	c := managedIdentityClient{id: options.ID, endpoint: imdsEndpoint, msiType: msiTypeIMDS}
	env := "IMDS"
	if endpoint, ok := os.LookupEnv(identityEndpoint); ok {
		if _, ok := os.LookupEnv(identityHeader); ok {
			if _, ok := os.LookupEnv(identityServerThumbprint); ok {
				env = "Service Fabric"
				c.endpoint = endpoint
				c.msiType = msiTypeServiceFabric
			} else {
				env = "App Service"
				c.endpoint = endpoint
				c.msiType = msiTypeAppService
			}
		} else if _, ok := os.LookupEnv(arcIMDSEndpoint); ok {
			env = "Azure Arc"
			c.endpoint = endpoint
			c.msiType = msiTypeAzureArc
		}
	} else if endpoint, ok := os.LookupEnv(msiEndpoint); ok {
		env = "Cloud Shell"
		c.endpoint = endpoint
		c.msiType = msiTypeCloudShell
	} else {
		setIMDSRetryOptionDefaults(&cp.Retry)
	}
	c.pipeline = runtime.NewPipeline(component, version, runtime.PipelineOptions{}, &cp)

	if log.Should(EventAuthentication) {
		log.Writef(EventAuthentication, "Managed Identity Credential will use %s managed identity", env)
	}

	return &c, nil
}

// provideToken acquires a token for MSAL's confidential.Client, which caches the token
func (c *managedIdentityClient) provideToken(ctx context.Context, params confidential.TokenProviderParameters) (confidential.TokenProviderResult, error) {
	result := confidential.TokenProviderResult{}
	tk, err := c.authenticate(ctx, c.id, params.Scopes)
	if err == nil {
		result.AccessToken = tk.Token
		result.ExpiresInSeconds = int(time.Until(tk.ExpiresOn).Seconds())
	}
	return result, err
}

// authenticate acquires an access token
func (c *managedIdentityClient) authenticate(ctx context.Context, id ManagedIDKind, scopes []string) (azcore.AccessToken, error) {
	var cancel context.CancelFunc
	if c.imdsTimeout > 0 && c.msiType == msiTypeIMDS {
		ctx, cancel = context.WithTimeout(ctx, c.imdsTimeout)
		defer cancel()
	}

	msg, err := c.createAuthRequest(ctx, id, scopes)
	if err != nil {
		return azcore.AccessToken{}, err
	}

	resp, err := c.pipeline.Do(msg)
	if err != nil {
		if cancel != nil && errors.Is(err, context.DeadlineExceeded) {
			return azcore.AccessToken{}, newCredentialUnavailableError(credNameManagedIdentity, "IMDS token request timed out")
		}
		return azcore.AccessToken{}, newAuthenticationFailedError(credNameManagedIdentity, err.Error(), nil)
	}

	// got a response, remove the IMDS timeout so future requests use the transport's configuration
	c.imdsTimeout = 0

	if runtime.HasStatusCode(resp, http.StatusOK, http.StatusCreated) {
		return c.createAccessToken(resp)
	}

	if c.msiType == msiTypeIMDS && resp.StatusCode == 400 {
		if id != nil {
			return azcore.AccessToken{}, newAuthenticationFailedError(credNameManagedIdentity, "the requested identity isn't assigned to this resource", resp)
		}
		return azcore.AccessToken{}, newCredentialUnavailableError(credNameManagedIdentity, "no default identity is assigned to this resource")
	}

	return azcore.AccessToken{}, newAuthenticationFailedError(credNameManagedIdentity, "authentication failed", resp)
}

func (c *managedIdentityClient) createAccessToken(res *http.Response) (azcore.AccessToken, error) {
	value := struct {
		// these are the only fields that we use
		Token        string        `json:"access_token,omitempty"`
		RefreshToken string        `json:"refresh_token,omitempty"`
		ExpiresIn    wrappedNumber `json:"expires_in,omitempty"` // this field should always return the number of seconds for which a token is valid
		ExpiresOn    interface{}   `json:"expires_on,omitempty"` // the value returned in this field varies between a number and a date string
	}{}
	if err := runtime.UnmarshalAsJSON(res, &value); err != nil {
		return azcore.AccessToken{}, fmt.Errorf("internal AccessToken: %v", err)
	}
	if value.ExpiresIn != "" {
		expiresIn, err := json.Number(value.ExpiresIn).Int64()
		if err != nil {
			return azcore.AccessToken{}, err
		}
		return azcore.AccessToken{Token: value.Token, ExpiresOn: time.Now().Add(time.Second * time.Duration(expiresIn)).UTC()}, nil
	}
	switch v := value.ExpiresOn.(type) {
	case float64:
		return azcore.AccessToken{Token: value.Token, ExpiresOn: time.Unix(int64(v), 0).UTC()}, nil
	case string:
		if expiresOn, err := strconv.Atoi(v); err == nil {
			return azcore.AccessToken{Token: value.Token, ExpiresOn: time.Unix(int64(expiresOn), 0).UTC()}, nil
		}
		return azcore.AccessToken{}, newAuthenticationFailedError(credNameManagedIdentity, "unexpected expires_on value: "+v, res)
	default:
		msg := fmt.Sprintf("unsupported type received in expires_on: %T, %v", v, v)
		return azcore.AccessToken{}, newAuthenticationFailedError(credNameManagedIdentity, msg, res)
	}
}

func (c *managedIdentityClient) createAuthRequest(ctx context.Context, id ManagedIDKind, scopes []string) (*policy.Request, error) {
	switch c.msiType {
	case msiTypeIMDS:
		return c.createIMDSAuthRequest(ctx, id, scopes)
	case msiTypeAppService:
		return c.createAppServiceAuthRequest(ctx, id, scopes)
	case msiTypeAzureArc:
		// need to perform preliminary request to retreive the secret key challenge provided by the HIMDS service
		key, err := c.getAzureArcSecretKey(ctx, scopes)
		if err != nil {
			msg := fmt.Sprintf("failed to retreive secret key from the identity endpoint: %v", err)
			return nil, newAuthenticationFailedError(credNameManagedIdentity, msg, nil)
		}
		return c.createAzureArcAuthRequest(ctx, id, scopes, key)
	case msiTypeServiceFabric:
		return c.createServiceFabricAuthRequest(ctx, id, scopes)
	case msiTypeCloudShell:
		return c.createCloudShellAuthRequest(ctx, id, scopes)
	default:
		return nil, newCredentialUnavailableError(credNameManagedIdentity, "managed identity isn't supported in this environment")
	}
}

func (c *managedIdentityClient) createIMDSAuthRequest(ctx context.Context, id ManagedIDKind, scopes []string) (*policy.Request, error) {
	request, err := runtime.NewRequest(ctx, http.MethodGet, c.endpoint)
	if err != nil {
		return nil, err
	}
	request.Raw().Header.Set(headerMetadata, "true")
	q := request.Raw().URL.Query()
	q.Add("api-version", imdsAPIVersion)
	q.Add("resource", strings.Join(scopes, " "))
	if id != nil {
		if id.idKind() == miResourceID {
			q.Add(qpResID, id.String())
		} else {
			q.Add(qpClientID, id.String())
		}
	}
	request.Raw().URL.RawQuery = q.Encode()
	return request, nil
}

func (c *managedIdentityClient) createAppServiceAuthRequest(ctx context.Context, id ManagedIDKind, scopes []string) (*policy.Request, error) {
	request, err := runtime.NewRequest(ctx, http.MethodGet, c.endpoint)
	if err != nil {
		return nil, err
	}
	request.Raw().Header.Set("X-IDENTITY-HEADER", os.Getenv(identityHeader))
	q := request.Raw().URL.Query()
	q.Add("api-version", "2019-08-01")
	q.Add("resource", scopes[0])
	if id != nil {
		if id.idKind() == miResourceID {
			q.Add(qpResID, id.String())
		} else {
			q.Add(qpClientID, id.String())
		}
	}
	request.Raw().URL.RawQuery = q.Encode()
	return request, nil
}

func (c *managedIdentityClient) createServiceFabricAuthRequest(ctx context.Context, id ManagedIDKind, scopes []string) (*policy.Request, error) {
	request, err := runtime.NewRequest(ctx, http.MethodGet, c.endpoint)
	if err != nil {
		return nil, err
	}
	q := request.Raw().URL.Query()
	request.Raw().Header.Set("Accept", "application/json")
	request.Raw().Header.Set("Secret", os.Getenv(identityHeader))
	q.Add("api-version", serviceFabricAPIVersion)
	q.Add("resource", strings.Join(scopes, " "))
	if id != nil {
		log.Write(EventAuthentication, "WARNING: Service Fabric doesn't support selecting a user-assigned identity at runtime")
		if id.idKind() == miResourceID {
			q.Add(qpResID, id.String())
		} else {
			q.Add(qpClientID, id.String())
		}
	}
	request.Raw().URL.RawQuery = q.Encode()
	return request, nil
}

func (c *managedIdentityClient) getAzureArcSecretKey(ctx context.Context, resources []string) (string, error) {
	// create the request to retreive the secret key challenge provided by the HIMDS service
	request, err := runtime.NewRequest(ctx, http.MethodGet, c.endpoint)
	if err != nil {
		return "", err
	}
	request.Raw().Header.Set(headerMetadata, "true")
	q := request.Raw().URL.Query()
	q.Add("api-version", azureArcAPIVersion)
	q.Add("resource", strings.Join(resources, " "))
	request.Raw().URL.RawQuery = q.Encode()
	// send the initial request to get the short-lived secret key
	response, err := c.pipeline.Do(request)
	if err != nil {
		return "", err
	}
	// the endpoint is expected to return a 401 with the WWW-Authenticate header set to the location
	// of the secret key file. Any other status code indicates an error in the request.
	if response.StatusCode != 401 {
		msg := fmt.Sprintf("expected a 401 response, received %d", response.StatusCode)
		return "", newAuthenticationFailedError(credNameManagedIdentity, msg, response)
	}
	header := response.Header.Get("WWW-Authenticate")
	if len(header) == 0 {
		return "", errors.New("did not receive a value from WWW-Authenticate header")
	}
	// the WWW-Authenticate header is expected in the following format: Basic realm=/some/file/path.key
	pos := strings.LastIndex(header, "=")
	if pos == -1 {
		return "", fmt.Errorf("did not receive a correct value from WWW-Authenticate header: %s", header)
	}
	key, err := os.ReadFile(header[pos+1:])
	if err != nil {
		return "", fmt.Errorf("could not read file (%s) contents: %v", header[pos+1:], err)
	}
	return string(key), nil
}

func (c *managedIdentityClient) createAzureArcAuthRequest(ctx context.Context, id ManagedIDKind, resources []string, key string) (*policy.Request, error) {
	request, err := runtime.NewRequest(ctx, http.MethodGet, c.endpoint)
	if err != nil {
		return nil, err
	}
	request.Raw().Header.Set(headerMetadata, "true")
	request.Raw().Header.Set("Authorization", fmt.Sprintf("Basic %s", key))
	q := request.Raw().URL.Query()
	q.Add("api-version", azureArcAPIVersion)
	q.Add("resource", strings.Join(resources, " "))
	if id != nil {
		log.Write(EventAuthentication, "WARNING: Azure Arc doesn't support user-assigned managed identities")
		if id.idKind() == miResourceID {
			q.Add(qpResID, id.String())
		} else {
			q.Add(qpClientID, id.String())
		}
	}
	request.Raw().URL.RawQuery = q.Encode()
	return request, nil
}

func (c *managedIdentityClient) createCloudShellAuthRequest(ctx context.Context, id ManagedIDKind, scopes []string) (*policy.Request, error) {
	request, err := runtime.NewRequest(ctx, http.MethodPost, c.endpoint)
	if err != nil {
		return nil, err
	}
	request.Raw().Header.Set(headerMetadata, "true")
	data := url.Values{}
	data.Set("resource", strings.Join(scopes, " "))
	dataEncoded := data.Encode()
	body := streaming.NopCloser(strings.NewReader(dataEncoded))
	if err := request.SetBody(body, "application/x-www-form-urlencoded"); err != nil {
		return nil, err
	}
	if id != nil {
		log.Write(EventAuthentication, "WARNING: Cloud Shell doesn't support user-assigned managed identities")
		q := request.Raw().URL.Query()
		if id.idKind() == miResourceID {
			q.Add(qpResID, id.String())
		} else {
			q.Add(qpClientID, id.String())
		}
	}
	return request, nil
}

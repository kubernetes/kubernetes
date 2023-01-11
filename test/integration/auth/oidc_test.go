package auth

import (
	"context"
	"net/http"
	"net/http/httputil"
	"strings"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	kubeoptions "k8s.io/kubernetes/pkg/kubeapiserver/options"
	"k8s.io/kubernetes/test/integration/auth/oidcserver"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestOIDCClients(t *testing.T) {
	oidcServer := oidcserver.RunOIDCMockServer(t)

	_, kubeConfig, tearDownFn := framework.StartTestServer(t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.Authentication.Anonymous = &kubeoptions.AnonymousAuthenticationOptions{Allow: false}
			opts.Authentication.APIAudiences = []string{oidcserver.KubeAudience}
			opts.Authentication.OIDC = oidcServer.OIDCConfig()
			opts.Authorization.Modes = []string{"AlwaysAllow"}
			opts.Authentication.TokenSuccessCacheTTL = -1
			opts.Authentication.TokenFailureCacheTTL = -1
		},
	})
	defer tearDownFn()

	// FIXME: should do at least two rountrips to retrieve the id-token in order
	// to attempt a token refresh with a new refresh-token (refresh token is currently
	// rotated with every /token request)
	for _, tt := range []struct {
		name                     string
		configToken              oidcserver.TokensType
		serverToken              oidcserver.TokensType
		expectUnauthorized       bool
		expectTokenRetrieveError bool
	}{
		{
			name:        "no init token, valid token from server",
			configToken: oidcserver.TokensNone,
			serverToken: oidcserver.TokensValid,
		},
		{
			name:               "no init token, expired token from server",
			configToken:        oidcserver.TokensNone,
			serverToken:        oidcserver.TokensExpired,
			expectUnauthorized: true,
		},
		{
			name:               "no init token, improperly signed token from server",
			configToken:        oidcserver.TokensNone,
			serverToken:        oidcserver.TokensInvalidSignature,
			expectUnauthorized: true,
		},
		{
			name:                     "no init token, no id-token from the server",
			configToken:              oidcserver.TokensNone,
			serverToken:              oidcserver.TokensNone,
			expectTokenRetrieveError: true, // TODO: if the received token is empty (id_token: "" in token response, not `omitempty`` on id_token, the empty string is accepted as the token - and rightfully fails with Unauthorized afterwards)
		},
		{
			name:        "valid init token",
			configToken: oidcserver.TokensValid,
			serverToken: oidcserver.TokensValid,
		},
		{
			name:        "expired init token, valid token from server",
			configToken: oidcserver.TokensExpired,
			serverToken: oidcserver.TokensValid,
		},
		{
			name:        "improperly signed init token, valid token from server",
			configToken: oidcserver.TokensInvalidSignature,
			serverToken: oidcserver.TokensValid,
			// TODO: since we've got the refresh token - should the client try to fix our id-token instead of failing?
			expectUnauthorized: true,
		},
	} {
		t.Run(tt.name, func(t *testing.T) {
			expectError := tt.expectTokenRetrieveError || tt.expectUnauthorized

			configToken, err := oidcServer.MintIDToken(tt.configToken)
			if err != nil {
				t.Fatalf("failed to create token: %v", err)
			}
			oidcServer.SetMintedTokensType(tt.serverToken)

			userConfig := rest.AnonymousClientConfig(kubeConfig)
			userConfig.AuthProvider = withIDToken(oidcServer.AuthConfig(), configToken)
			recorder := httprecorder{t: t}
			userConfig.Wrap(recorder.Wrap)

			userClient, err := kubernetes.NewForConfig(userConfig)
			if err != nil {
				t.Fatalf("failed to setup user client: %v", err)
			}

			_, err = userClient.CoreV1().Namespaces().List(context.Background(), metav1.ListOptions{})
			if (err != nil) != expectError {
				t.Errorf("expected unauthorized (%v) or token retrieval (%v), got %v", tt.expectUnauthorized, tt.expectTokenRetrieveError, err)
			}

			if err != nil {
				if tt.expectUnauthorized && !apierrors.IsUnauthorized(err) {
					t.Errorf("expected unauthorized err, got %v", err)
				}

				if tt.expectTokenRetrieveError &&
					!strings.Contains(err.Error(),
						"token response did not contain an id_token, either the scope \"openid\" wasn't requested upon login, or the provider doesn't support id_tokens as part of the refresh response",
					) {
					t.Errorf("expected token retrieval error, got %v", err)
				}
			}
		})
	}
}

type httprecorder struct {
	delegate http.RoundTripper
	t        *testing.T
}

func (r *httprecorder) RoundTrip(req *http.Request) (*http.Response, error) {
	reqDump, _ := httputil.DumpRequestOut(req, false)
	r.t.Logf("request:\n%s", reqDump)
	resp, err := r.delegate.RoundTrip(req)
	respDump, _ := httputil.DumpResponse(resp, false)
	r.t.Logf("response:\n%s", respDump)
	return resp, err
}

func (r *httprecorder) Wrap(rt http.RoundTripper) http.RoundTripper {
	r.delegate = rt
	return r
}

func withIDToken(authConfig *clientcmdapi.AuthProviderConfig, token string) *clientcmdapi.AuthProviderConfig {
	authConfig.Config["id-token"] = token
	return authConfig
}

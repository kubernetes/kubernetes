/*
Copyright 2023 The Kubernetes Authors.

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
	"crypto/rsa"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	_ "k8s.io/client-go/plugin/pkg/client/auth/oidc"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd/api"
	certutil "k8s.io/client-go/util/cert"
	kubeapiserverapptesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/kubeapiserver/authorizer/modes"
	"k8s.io/kubernetes/test/integration/framework"
	utilsoidc "k8s.io/kubernetes/test/utils/oidc"
	utilsnet "k8s.io/utils/net"
)

const (
	defaultNamespace           = "default"
	defaultOIDCClientID        = "f403b682-603f-4ec9-b3e4-cf111ef36f7c"
	defaultOIDCClaimedUsername = "john_doe"
	defaultOIDCUsernamePrefix  = "k8s-"
	defaultRBACRoleName        = "developer-role"
	defaultRBACRoleBindingName = "developer-role-binding"

	defaultStubRefreshToken = "_fake_refresh_token_"
	defaultStubAccessToken  = "_fake_access_token_"

	rsaKeyBitSize = 2048
)

var (
	defaultRole = &rbacv1.Role{
		TypeMeta:   metav1.TypeMeta{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "Role"},
		ObjectMeta: metav1.ObjectMeta{Name: defaultRBACRoleName},
		Rules: []rbacv1.PolicyRule{
			{
				Verbs:         []string{"list"},
				Resources:     []string{"pods"},
				APIGroups:     []string{""},
				ResourceNames: []string{},
			},
		},
	}
	defaultRoleBinding = &rbacv1.RoleBinding{
		TypeMeta:   metav1.TypeMeta{APIVersion: "rbac.authorization.k8s.io/v1", Kind: "RoleBinding"},
		ObjectMeta: metav1.ObjectMeta{Name: defaultRBACRoleBindingName},
		Subjects: []rbacv1.Subject{
			{
				APIGroup: rbac.GroupName,
				Kind:     rbacv1.UserKind,
				Name:     defaultOIDCUsernamePrefix + defaultOIDCClaimedUsername,
			},
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: rbac.GroupName,
			Kind:     "Role",
			Name:     defaultRBACRoleName,
		},
	}
)

func TestOIDC(t *testing.T) {
	var tests = []struct {
		name                    string
		configureInfrastructure func(t *testing.T) (
			oidcServer *utilsoidc.TestServer,
			apiServer *kubeapiserverapptesting.TestServer,
			signingPrivateKey *rsa.PrivateKey,
			caCertContent []byte,
			caFilePath string,
		)
		configureOIDCServerBehaviour func(t *testing.T, oidcServer *utilsoidc.TestServer, signingPrivateKey *rsa.PrivateKey)
		configureClient              func(
			t *testing.T,
			restCfg *rest.Config,
			caCert []byte,
			certPath,
			oidcServerURL,
			oidcServerTokenURL string,
		) *kubernetes.Clientset
		asserErrFn func(t *testing.T, errorToCheck error)
	}{
		{
			name:                    "ID token is ok",
			configureInfrastructure: configureTestInfrastructure,
			configureOIDCServerBehaviour: func(t *testing.T, oidcServer *utilsoidc.TestServer, signingPrivateKey *rsa.PrivateKey) {
				idTokenLifetime := time.Second * 1200
				oidcServer.TokenHandler().EXPECT().Token().Times(1).DoAndReturn(utilsoidc.TokenHandlerBehaviourReturningPredefinedJWT(
					t,
					signingPrivateKey,
					oidcServer.URL(),
					defaultOIDCClientID,
					defaultOIDCClaimedUsername,
					defaultStubAccessToken,
					defaultStubRefreshToken,
					time.Now().Add(idTokenLifetime).Unix(),
				))
			},
			configureClient: configureClientFetchingOIDCCredentials,
			asserErrFn: func(t *testing.T, errorToCheck error) {
				assert.NoError(t, errorToCheck)
			},
		},
		{
			name:                    "ID token is expired",
			configureInfrastructure: configureTestInfrastructure,
			configureOIDCServerBehaviour: func(t *testing.T, oidcServer *utilsoidc.TestServer, signingPrivateKey *rsa.PrivateKey) {
				configureOIDCServerToReturnExpiredIDToken(t, 2, oidcServer, signingPrivateKey)
			},
			configureClient: configureClientFetchingOIDCCredentials,
			asserErrFn: func(t *testing.T, errorToCheck error) {
				assert.True(t, apierrors.IsUnauthorized(errorToCheck), errorToCheck)
			},
		},
		{
			name:                    "wrong client ID",
			configureInfrastructure: configureTestInfrastructure,
			configureOIDCServerBehaviour: func(t *testing.T, oidcServer *utilsoidc.TestServer, _ *rsa.PrivateKey) {
				oidcServer.TokenHandler().EXPECT().Token().Times(2).Return(utilsoidc.Token{}, utilsoidc.ErrBadClientID)
			},
			configureClient: configureClientWithEmptyIDToken,
			asserErrFn: func(t *testing.T, errorToCheck error) {
				urlError, ok := errorToCheck.(*url.Error)
				require.True(t, ok)
				assert.Equal(
					t,
					"failed to refresh token: oauth2: cannot fetch token: 400 Bad Request\nResponse: client ID is bad\n",
					urlError.Err.Error(),
				)
			},
		},
		{
			name:                         "client has wrong CA",
			configureInfrastructure:      configureTestInfrastructure,
			configureOIDCServerBehaviour: func(t *testing.T, oidcServer *utilsoidc.TestServer, _ *rsa.PrivateKey) {},
			configureClient: func(t *testing.T, restCfg *rest.Config, caCert []byte, _, oidcServerURL, oidcServerTokenURL string) *kubernetes.Clientset {
				tempDir := t.TempDir()
				certFilePath := filepath.Join(tempDir, "localhost_127.0.0.1_.crt")

				_, _, wantErr := certutil.GenerateSelfSignedCertKeyWithFixtures("localhost", []net.IP{utilsnet.ParseIPSloppy("127.0.0.1")}, nil, tempDir)
				require.NoError(t, wantErr)

				return configureClientWithEmptyIDToken(t, restCfg, caCert, certFilePath, oidcServerURL, oidcServerTokenURL)
			},
			asserErrFn: func(t *testing.T, errorToCheck error) {
				expectedErr := new(x509.UnknownAuthorityError)
				assert.ErrorAs(t, errorToCheck, expectedErr)
			},
		},
		{
			name:                    "refresh flow does not return ID Token",
			configureInfrastructure: configureTestInfrastructure,
			configureOIDCServerBehaviour: func(t *testing.T, oidcServer *utilsoidc.TestServer, signingPrivateKey *rsa.PrivateKey) {
				configureOIDCServerToReturnExpiredIDToken(t, 1, oidcServer, signingPrivateKey)
				oidcServer.TokenHandler().EXPECT().Token().Times(1).Return(utilsoidc.Token{
					IDToken:      "",
					AccessToken:  defaultStubAccessToken,
					RefreshToken: defaultStubRefreshToken,
					ExpiresIn:    time.Now().Add(time.Second * 1200).Unix(),
				}, nil)
			},
			configureClient: configureClientFetchingOIDCCredentials,
			asserErrFn: func(t *testing.T, errorToCheck error) {
				expectedError := new(apierrors.StatusError)
				assert.ErrorAs(t, errorToCheck, &expectedError)
				assert.Equal(
					t,
					`pods is forbidden: User "system:anonymous" cannot list resource "pods" in API group "" in the namespace "default"`,
					errorToCheck.Error(),
				)
			},
		},
		{
			name: "ID token signature can not be verified due to wrong JWKs",
			configureInfrastructure: func(t *testing.T) (
				oidcServer *utilsoidc.TestServer,
				apiServer *kubeapiserverapptesting.TestServer,
				signingPrivateKey *rsa.PrivateKey,
				caCertContent []byte,
				caFilePath string,
			) {
				caCertContent, _, caFilePath, caKeyFilePath := generateCert(t)

				signingPrivateKey, wantErr := rsa.GenerateKey(rand.Reader, rsaKeyBitSize)
				require.NoError(t, wantErr)

				oidcServer = utilsoidc.BuildAndRunTestServer(t, caFilePath, caKeyFilePath)
				apiServer = startTestAPIServerForOIDC(t, oidcServer.URL(), defaultOIDCClientID, caFilePath)

				adminClient := kubernetes.NewForConfigOrDie(apiServer.ClientConfig)
				configureRBAC(t, adminClient, defaultRole, defaultRoleBinding)

				anotherSigningPrivateKey, wantErr := rsa.GenerateKey(rand.Reader, rsaKeyBitSize)
				require.NoError(t, wantErr)
				oidcServer.JwksHandler().EXPECT().KeySet().AnyTimes().DoAndReturn(utilsoidc.DefaultJwksHandlerBehaviour(t, &anotherSigningPrivateKey.PublicKey))

				return oidcServer, apiServer, signingPrivateKey, caCertContent, caFilePath
			},
			configureOIDCServerBehaviour: func(t *testing.T, oidcServer *utilsoidc.TestServer, signingPrivateKey *rsa.PrivateKey) {
				oidcServer.TokenHandler().EXPECT().Token().Times(1).DoAndReturn(utilsoidc.TokenHandlerBehaviourReturningPredefinedJWT(
					t,
					signingPrivateKey,
					oidcServer.URL(),
					defaultOIDCClientID,
					defaultOIDCClaimedUsername,
					defaultStubAccessToken,
					defaultStubRefreshToken,
					time.Now().Add(time.Second*1200).Unix(),
				))
			},
			configureClient: configureClientFetchingOIDCCredentials,
			asserErrFn: func(t *testing.T, errorToCheck error) {
				assert.True(t, apierrors.IsUnauthorized(errorToCheck), errorToCheck)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			oidcServer, apiServer, signingPrivateKey, caCert, certPath := tt.configureInfrastructure(t)

			tt.configureOIDCServerBehaviour(t, oidcServer, signingPrivateKey)

			tokenURL, err := oidcServer.TokenURL()
			require.NoError(t, err)

			client := tt.configureClient(t, apiServer.ClientConfig, caCert, certPath, oidcServer.URL(), tokenURL)

			ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
			defer cancel()

			_, err = client.CoreV1().Pods(defaultNamespace).List(ctx, metav1.ListOptions{})

			tt.asserErrFn(t, err)
		})
	}
}

func TestUpdatingRefreshTokenInCaseOfExpiredIDToken(t *testing.T) {
	var tests = []struct {
		name                            string
		configureUpdatingTokenBehaviour func(t *testing.T, oidcServer *utilsoidc.TestServer, signingPrivateKey *rsa.PrivateKey)
		asserErrFn                      func(t *testing.T, errorToCheck error)
	}{
		{
			name: "cache returns stale client if refresh token is not updated in config",
			configureUpdatingTokenBehaviour: func(t *testing.T, oidcServer *utilsoidc.TestServer, signingPrivateKey *rsa.PrivateKey) {
				oidcServer.TokenHandler().EXPECT().Token().Times(1).DoAndReturn(utilsoidc.TokenHandlerBehaviourReturningPredefinedJWT(
					t,
					signingPrivateKey,
					oidcServer.URL(),
					defaultOIDCClientID,
					defaultOIDCClaimedUsername,
					defaultStubAccessToken,
					defaultStubRefreshToken,
					time.Now().Add(time.Second*1200).Unix(),
				))
				configureOIDCServerToReturnExpiredRefreshTokenErrorOnTryingToUpdateIDToken(oidcServer)
			},
			asserErrFn: func(t *testing.T, errorToCheck error) {
				urlError, ok := errorToCheck.(*url.Error)
				require.True(t, ok)
				assert.Equal(
					t,
					"failed to refresh token: oauth2: cannot fetch token: 400 Bad Request\nResponse: refresh token is expired\n",
					urlError.Err.Error(),
				)
			},
		},
	}

	oidcServer, apiServer, signingPrivateKey, caCert, certPath := configureTestInfrastructure(t)

	tokenURL, err := oidcServer.TokenURL()
	require.NoError(t, err)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			expiredIDToken, stubRefreshToken := fetchExpiredToken(t, oidcServer, caCert, signingPrivateKey)
			clientConfig := configureClientConfigForOIDC(t, apiServer.ClientConfig, defaultOIDCClientID, certPath, expiredIDToken, stubRefreshToken, oidcServer.URL())
			expiredClient := kubernetes.NewForConfigOrDie(clientConfig)
			configureOIDCServerToReturnExpiredRefreshTokenErrorOnTryingToUpdateIDToken(oidcServer)

			ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
			defer cancel()

			_, err = expiredClient.CoreV1().Pods(defaultNamespace).List(ctx, metav1.ListOptions{})
			assert.Error(t, err)

			tt.configureUpdatingTokenBehaviour(t, oidcServer, signingPrivateKey)
			idToken, stubRefreshToken := fetchOIDCCredentials(t, tokenURL, caCert)
			clientConfig = configureClientConfigForOIDC(t, apiServer.ClientConfig, defaultOIDCClientID, certPath, idToken, stubRefreshToken, oidcServer.URL())
			expectedOkClient := kubernetes.NewForConfigOrDie(clientConfig)
			_, err = expectedOkClient.CoreV1().Pods(defaultNamespace).List(ctx, metav1.ListOptions{})

			tt.asserErrFn(t, err)
		})
	}
}

func configureTestInfrastructure(t *testing.T) (
	oidcServer *utilsoidc.TestServer,
	apiServer *kubeapiserverapptesting.TestServer,
	signingPrivateKey *rsa.PrivateKey,
	caCertContent []byte,
	caFilePath string,
) {
	t.Helper()

	caCertContent, _, caFilePath, caKeyFilePath := generateCert(t)

	signingPrivateKey, err := rsa.GenerateKey(rand.Reader, rsaKeyBitSize)
	require.NoError(t, err)

	oidcServer = utilsoidc.BuildAndRunTestServer(t, caFilePath, caKeyFilePath)
	apiServer = startTestAPIServerForOIDC(t, oidcServer.URL(), defaultOIDCClientID, caFilePath)

	oidcServer.JwksHandler().EXPECT().KeySet().AnyTimes().DoAndReturn(utilsoidc.DefaultJwksHandlerBehaviour(t, &signingPrivateKey.PublicKey))

	adminClient := kubernetes.NewForConfigOrDie(apiServer.ClientConfig)
	configureRBAC(t, adminClient, defaultRole, defaultRoleBinding)

	return oidcServer, apiServer, signingPrivateKey, caCertContent, caFilePath
}

func configureClientFetchingOIDCCredentials(t *testing.T, restCfg *rest.Config, caCert []byte, certPath, oidcServerURL, oidcServerTokenURL string) *kubernetes.Clientset {
	idToken, stubRefreshToken := fetchOIDCCredentials(t, oidcServerTokenURL, caCert)
	clientConfig := configureClientConfigForOIDC(t, restCfg, defaultOIDCClientID, certPath, idToken, stubRefreshToken, oidcServerURL)
	return kubernetes.NewForConfigOrDie(clientConfig)
}

func configureClientWithEmptyIDToken(t *testing.T, restCfg *rest.Config, _ []byte, certPath, oidcServerURL, _ string) *kubernetes.Clientset {
	emptyIDToken, stubRefreshToken := "", defaultStubRefreshToken
	clientConfig := configureClientConfigForOIDC(t, restCfg, defaultOIDCClientID, certPath, emptyIDToken, stubRefreshToken, oidcServerURL)
	return kubernetes.NewForConfigOrDie(clientConfig)
}

func configureRBAC(t *testing.T, clientset *kubernetes.Clientset, role *rbacv1.Role, binding *rbacv1.RoleBinding) {
	t.Helper()

	ctx, cancel := context.WithTimeout(context.Background(), time.Second*30)
	defer cancel()

	_, err := clientset.RbacV1().Roles(defaultNamespace).Create(ctx, role, metav1.CreateOptions{})
	require.NoError(t, err)
	_, err = clientset.RbacV1().RoleBindings(defaultNamespace).Create(ctx, binding, metav1.CreateOptions{})
	require.NoError(t, err)
}

func configureClientConfigForOIDC(t *testing.T, config *rest.Config, clientID, caFilePath, idToken, refreshToken, oidcServerURL string) *rest.Config {
	t.Helper()
	cfg := rest.AnonymousClientConfig(config)
	cfg.AuthProvider = &api.AuthProviderConfig{
		Name: "oidc",
		Config: map[string]string{
			"client-id":                 clientID,
			"id-token":                  idToken,
			"idp-issuer-url":            oidcServerURL,
			"idp-certificate-authority": caFilePath,
			"refresh-token":             refreshToken,
		},
	}

	return cfg
}

func startTestAPIServerForOIDC(t *testing.T, oidcURL, oidcClientID, oidcCAFilePath string) *kubeapiserverapptesting.TestServer {
	t.Helper()

	server, err := kubeapiserverapptesting.StartTestServer(
		t,
		kubeapiserverapptesting.NewDefaultTestServerOptions(),
		[]string{
			fmt.Sprintf("--oidc-issuer-url=%s", oidcURL),
			fmt.Sprintf("--oidc-client-id=%s", oidcClientID),
			fmt.Sprintf("--oidc-ca-file=%s", oidcCAFilePath),
			fmt.Sprintf("--oidc-username-prefix=%s", defaultOIDCUsernamePrefix),
			fmt.Sprintf("--authorization-mode=%s", modes.ModeRBAC),
		},
		framework.SharedEtcd(),
	)
	require.NoError(t, err)

	t.Cleanup(server.TearDownFn)

	return &server
}

func fetchOIDCCredentials(t *testing.T, oidcTokenURL string, caCertContent []byte) (idToken, refreshToken string) {
	t.Helper()

	req, err := http.NewRequest(http.MethodGet, oidcTokenURL, http.NoBody)
	require.NoError(t, err)

	caPool := x509.NewCertPool()
	ok := caPool.AppendCertsFromPEM(caCertContent)
	require.True(t, ok)

	client := http.Client{Transport: &http.Transport{
		TLSClientConfig: &tls.Config{
			RootCAs: caPool,
		},
	}}

	token := new(utilsoidc.Token)

	resp, err := client.Do(req)
	require.NoError(t, err)

	err = json.NewDecoder(resp.Body).Decode(token)
	require.NoError(t, err)

	return token.IDToken, token.RefreshToken
}

func fetchExpiredToken(t *testing.T, oidcServer *utilsoidc.TestServer, caCertContent []byte, signingPrivateKey *rsa.PrivateKey) (expiredToken, stubRefreshToken string) {
	t.Helper()

	tokenURL, err := oidcServer.TokenURL()
	require.NoError(t, err)

	configureOIDCServerToReturnExpiredIDToken(t, 1, oidcServer, signingPrivateKey)
	expiredToken, stubRefreshToken = fetchOIDCCredentials(t, tokenURL, caCertContent)

	return expiredToken, stubRefreshToken
}

func configureOIDCServerToReturnExpiredIDToken(t *testing.T, returningExpiredTokenTimes int, oidcServer *utilsoidc.TestServer, signingPrivateKey *rsa.PrivateKey) {
	t.Helper()

	oidcServer.TokenHandler().EXPECT().Token().Times(returningExpiredTokenTimes).DoAndReturn(func() (utilsoidc.Token, error) {
		token, err := utilsoidc.TokenHandlerBehaviourReturningPredefinedJWT(
			t,
			signingPrivateKey,
			oidcServer.URL(),
			defaultOIDCClientID,
			defaultOIDCClaimedUsername,
			defaultStubAccessToken,
			defaultStubRefreshToken,
			time.Now().Add(-time.Millisecond).Unix(),
		)()
		return token, err
	})
}

func configureOIDCServerToReturnExpiredRefreshTokenErrorOnTryingToUpdateIDToken(oidcServer *utilsoidc.TestServer) {
	oidcServer.TokenHandler().EXPECT().Token().Times(2).Return(utilsoidc.Token{}, utilsoidc.ErrRefreshTokenExpired)
}

func generateCert(t *testing.T) (cert, key []byte, certFilePath, keyFilePath string) {
	t.Helper()

	tempDir := t.TempDir()
	certFilePath = filepath.Join(tempDir, "localhost_127.0.0.1_.crt")
	keyFilePath = filepath.Join(tempDir, "localhost_127.0.0.1_.key")

	cert, key, err := certutil.GenerateSelfSignedCertKeyWithFixtures("localhost", []net.IP{utilsnet.ParseIPSloppy("127.0.0.1")}, nil, tempDir)
	require.NoError(t, err)

	return cert, key, certFilePath, keyFilePath
}

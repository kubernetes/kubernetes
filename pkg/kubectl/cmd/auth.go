/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
package cmd

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"

	"github.com/spf13/cobra"
	"k8s.io/kubernetes/pkg/client/restclient"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	cmdconfig "k8s.io/kubernetes/pkg/kubectl/cmd/config"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/plugin/pkg/auth/authenticator/token/oidc"
)

// NewCmdAuth creates a command object for the generic "auth" action, which
// performs various authenitcation subcommands.
func NewCmdAuth(f *cmdutil.Factory, configAccess cmdconfig.ConfigAccess, out io.Writer) *cobra.Command {

	cmd := &cobra.Command{
		Use:   "auth SUBCOMMAND",
		Short: "auth performs various various authentication related commands.",
		Long: `auth performs various various authentication related commands like "kubectl auth refresh"

These commands may alter your kubeconfig file, subject to the same loading order as the various "kubectl config" commands"
`,

		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}

	cmd.AddCommand(NewCmdAuthRefresh(out, f, configAccess))
	return cmd
}

// NewCmdAuthRefresh creates a command object for the "auth refresh" command which refreshes an OIDC id_token using a refresh token stored in the kube config.
func NewCmdAuthRefresh(out io.Writer, f *cmdutil.Factory, configAccess cmdconfig.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "refresh",
		Short: "refresh gets an authentication token from the stored refresh token. ",
		Long: `refresh uses the refresh token stored in the kubeconfig to obtain an authentication token; the obtained token is then stored in the kubeconfig.

For this command to work, "current-context" must be set in the kubeconfig, and that context must have an "oidc" section with a valid "refresh-token" inside of it. 
`,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunAuthRefresh(out, cmd, f, configAccess, args)
			cmdutil.CheckErr(err)
		},
	}
	return cmd
}

func RunAuthRefresh(out io.Writer, cmd *cobra.Command, f *cmdutil.Factory, configAccess cmdconfig.ConfigAccess, args []string) error {
	cfg, err := configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	// Make sure there's a user.
	user, err := GetCurrentAuthInfo(cfg)
	if err != nil {
		return err
	}

	// Ensure that the AuthInfo has an OIDC
	if user.OIDCInfo == nil {
		return fmt.Errorf("The current-context's user has no oidc information")
	}

	// Ensure that the OIDC has a refresh token
	if user.OIDCInfo.RefreshToken == "" {
		return errors.New("The current-context's user has no oidc refresh token")
	}

	// Ensure that a cluster is set.
	cluster, err := GetCurrentCluster(cfg)
	if err != nil {
		return err
	}

	authClient, err := NewOIDCAuthClient(cluster)
	if err != nil {
		return err
	}

	// Do the exchange.
	idToken, err := authClient.ExchangeRefreshToken(user.OIDCInfo.RefreshToken)
	if err != nil {
		return err
	}
	if idToken == "" {
		return errors.New("No id_token returned from API Server")
	}

	// Store the obtained token in the config.
	user.Token = idToken
	err = cmdconfig.ModifyConfig(configAccess, *cfg, false)
	if err != nil {
		return err
	}
	return nil
}

func GetCurrentContext(cfg *clientcmdapi.Config) (*clientcmdapi.Context, error) {
	// Determine if there is CurrentContext set
	if cfg.CurrentContext == "" {
		return nil, fmt.Errorf("current-context must be set in config file")
	}

	// Get the current context
	ctx, ok := cfg.Contexts[cfg.CurrentContext]
	if !ok {
		return nil, fmt.Errorf("context %q does not exist", cfg.CurrentContext)
	}

	return ctx, nil
}

func GetCurrentAuthInfo(cfg *clientcmdapi.Config) (*clientcmdapi.AuthInfo, error) {
	ctx, err := GetCurrentContext(cfg)
	if err != nil {
		return nil, err
	}

	// Ensure that it has an AuthInfo
	if ctx.AuthInfo == "" {
		return nil, fmt.Errorf("the current-context must have a user set")
	}
	user, ok := cfg.AuthInfos[ctx.AuthInfo]
	if !ok {
		return nil, fmt.Errorf("user %q does not exist in the configuration", ctx.AuthInfo)
	}
	return user, nil
}

func GetCurrentCluster(cfg *clientcmdapi.Config) (*clientcmdapi.Cluster, error) {
	ctx, err := GetCurrentContext(cfg)
	if err != nil {
		return nil, err
	}

	if ctx.Cluster == "" {
		return nil, errors.New("the current-context must have a cluster set")
	}
	cluster, ok := cfg.Clusters[ctx.Cluster]
	if !ok {
		return nil, fmt.Errorf("cluster %q does not exist in the configuration", ctx.Cluster)
	}
	return cluster, nil
}

// OIDCAuthClient is an object performs operations related to OpenID Connect Authentication against the API Server.
type OIDCAuthClient struct {
	server     url.URL
	httpClient *http.Client
}

// NewOIDCAuthClient creates a new OIDCAuthClient for the given cluster.
func NewOIDCAuthClient(cluster *clientcmdapi.Cluster) (*OIDCAuthClient, error) {
	if cluster.Server == "" {
		return nil, fmt.Errorf("cluster has no server set")
	}

	srvURL, err := url.Parse(cluster.Server)
	if err != nil {
		return nil, err
	}

	httpClient, err := HTTPClientForCluster(cluster)
	if err != nil {
		return nil, err
	}

	return &OIDCAuthClient{
		server:     *srvURL,
		httpClient: httpClient,
	}, nil
}

// ExchangeRefreshToken exchanges the given refresh token for an ID token.
func (o *OIDCAuthClient) ExchangeRefreshToken(refreshToken string) (string, error) {
	reqURL := o.server
	reqURL.Path = oidc.PathExchangeRefreshToken
	q := url.Values{}
	q.Set("refresh_token", refreshToken)
	req, err := http.NewRequest("POST", reqURL.String(),
		bytes.NewBuffer([]byte(q.Encode())))
	if err != nil {
		return "", err
	}
	req.Header.Add("Content-Type", "application/x-www-form-urlencoded")

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Non 200 status returned from server: %v", resp.StatusCode)
	}

	raw, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}

	q, err = url.ParseQuery(string(raw))
	if err != nil {
		return "", err
	}

	idToken := q.Get("id_token")
	if idToken == "" {
		return "", errors.New("Missing ID token from response")
	}

	return idToken, err
}

func HTTPClientForCluster(cluster *clientcmdapi.Cluster) (*http.Client, error) {
	clientConfig := restclient.Config{
		TLSClientConfig: restclient.TLSClientConfig{
			CAFile: cluster.CertificateAuthority,
			CAData: cluster.CertificateAuthorityData,
		},
	}

	trans, err := restclient.TransportFor(&clientConfig)
	if err != nil {
		return nil, err
	}

	httpClient := &http.Client{Transport: trans}
	return httpClient, nil
}

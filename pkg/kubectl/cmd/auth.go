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
	"net"
	"net/http"
	"net/url"
	"sync"

	"github.com/pkg/browser"
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
	cmd.AddCommand(NewCmdAuthLogin(out, f, configAccess))
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

// NewCmdAuthLogin creates a command object for the "auth login" command, which authenticates a user via the browser and stores the obtained token in the kubeconfig.
func NewCmdAuthLogin(out io.Writer, f *cmdutil.Factory, configAccess cmdconfig.ConfigAccess) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "login",
		Short: "login obtains authentication credentials to the cluster via the browser",
		Long: `login will spawn the system's default browser, and navigate it to a page which allow the user to authenticate and obtain user credentials for interacting with the cluster.

The authentication token will be stored in the current-context's user's "token" field. If a refresh-token is obtained, it will be stored in the user's oidc's section "refresh-token" field.
 `,
		Run: func(cmd *cobra.Command, args []string) {
			err := RunAuthLogin(out, cmd, f, configAccess, args)
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

func RunAuthLogin(out io.Writer, cmd *cobra.Command, f *cmdutil.Factory, configAccess cmdconfig.ConfigAccess, args []string) error {
	cfg, err := configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	cluster, err := GetCurrentCluster(cfg)
	if err != nil {
		return err
	}

	user, err := GetCurrentAuthInfo(cfg)
	if err != nil {
		return err
	}

	authClient, err := NewOIDCAuthClient(cluster)
	if err != nil {
		return err
	}

	// Get an idToken and refreshToken by having the user authenticate.
	fmt.Fprintf(out, "Waiting for in-browser authentication. Hit ctrl-c to exit.\n")
	idToken, refreshToken, err := authClient.Login()
	if err != nil {
		return err
	}

	// Update the Token and RefreshToken for the user.
	user.Token = idToken
	if refreshToken != "" {
		oidcInfo := user.OIDCInfo
		if oidcInfo == nil {
			user.OIDCInfo = &clientcmdapi.OIDCInfo{}
			oidcInfo = user.OIDCInfo
		}
		oidcInfo.RefreshToken = refreshToken
	}

	// Save changes to the user.
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

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("Server returned non-200 status: %d", resp.StatusCode)
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

// Login opens a browser for the user to authenticate to, and uses those credentials to obtain an ID token and refresh token, which are the return values (respectively) along with an error.
// This method also starts a web server listening on a random port on localhost
// to facilitate communication back from the browser.
func (o *OIDCAuthClient) Login() (string, string, error) {

	// Pick a random open port to listen on.
	listener, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", "", err
	}

	var wg sync.WaitGroup
	var code string
	var reqErr error
	// This server waits for the redirect coming back from API server, populates
	// code and reqErr from that request, and then stops itself.
	srv := &http.Server{
		Handler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// This is to handle unwanted but inevitable requests, like for
			// "favicon.ico"
			if r.URL.Path != "/" {
				return
			}

			// Stop listening once we've gotten a request.
			listener.Close()
			if r.Method != "GET" {
				reqErr = errors.New("The server made a bad request: Only GET is allowed")
			}

			code = r.URL.Query().Get("code")
			if code == "" {
				reqErr = errors.New("Missing 'code' parameter from server.")
			}

			var msg string
			if reqErr == nil {
				msg = "Login Successful!"
			} else {
				msg = reqErr.Error()
			}
			w.Write([]byte(fmt.Sprintf(authPostLoginTpl, msg)))
			wg.Done()
		}),
	}
	wg.Add(1)
	go srv.Serve(listener)

	// Construct the URL to the API Server's token exchange endpoint to open in
	// the browser, with the "callback" parameter set to the local server that
	// was started.
	reqURL := o.server
	reqURL.Path = oidc.PathAuthenticate
	localAddr := "http://" + listener.Addr().String()
	q := reqURL.Query()
	q.Set("callback", localAddr)
	reqURL.RawQuery = q.Encode()

	// Open the browser and wait for the callback.
	err = browser.OpenURL(reqURL.String())
	if err != nil {
		return "", "", err
	}
	wg.Wait()
	if reqErr != nil {
		return "", "", reqErr
	}

	// Exchange the code obtained from the callback redirect for an id and
	// refresh token.
	idToken, refreshToken, err := o.ExchangeAuthCode(code)
	if err != nil {
		return "", "", err
	}

	if idToken == "" {
		return "", "", errors.New("No ID Token returned from API Server")
	}

	return idToken, refreshToken, nil
}

// ExchangeAuthCode exchanges the authorization code for an id token and refresh token, which are returned (respectively) along with an error.
func (o *OIDCAuthClient) ExchangeAuthCode(code string) (string, string, error) {
	reqURL := o.server
	reqURL.Path = oidc.PathExchangeCode
	q := reqURL.Query()
	q.Set("code", code)
	req, err := http.NewRequest("POST", reqURL.String(),
		bytes.NewBuffer([]byte(q.Encode())))
	if err != nil {
		return "", "", err
	}

	req.Header.Add("Content-Type", "application/x-www-form-urlencoded")
	resp, err := o.httpClient.Do(req)
	if err != nil {
		return "", "", err
	}
	if resp.StatusCode != http.StatusOK {
		return "", "", fmt.Errorf("Server returned non-200 status: %d", resp.StatusCode)
	}

	raw, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", "", err
	}

	v, err := url.ParseQuery(string(raw))
	if err != nil {
		return "", "", err
	}

	return v.Get("id_token"), v.Get("refresh_token"), nil
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

const authPostLoginTpl = `
  <body>
    %v
    <br>
    You can now close this window.
  </body>
</html>`

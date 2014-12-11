/*
Copyright 2014 Google Inc. All rights reserved.

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

package clientcmd

import (
	"fmt"
	"os"
	"reflect"

	"github.com/spf13/pflag"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
)

// Builder are used to bind and interpret command line flags to make it easy to get an api server client
type Builder interface {
	// BindFlags must bind and keep track of all the flags required to build a client config object
	BindFlags(flags *pflag.FlagSet)
	// Config uses the values of the bound flags and builds a complete client config
	Config() (*client.Config, error)
	// Client calls BuildConfig under the covers and uses that config to return a client
	Client() (*client.Client, error)
}

// cmdAuthInfo is used to track whether flags have been set
type cmdAuthInfo struct {
	User        StringFlag
	Password    StringFlag
	CAFile      StringFlag
	CertFile    StringFlag
	KeyFile     StringFlag
	BearerToken StringFlag
	Insecure    BoolFlag
}

// builder is a default implementation of a Builder
type builder struct {
	authLoader      AuthLoader
	cmdAuthInfo     cmdAuthInfo
	authPath        string
	apiserver       string
	apiVersion      string
	matchApiVersion bool
}

// NewBuilder returns a valid Builder that uses the passed authLoader.  If authLoader is nil, the NewDefaultAuthLoader is used.
func NewBuilder(authLoader AuthLoader) Builder {
	if authLoader == nil {
		authLoader = NewDefaultAuthLoader()
	}

	return &builder{
		authLoader: authLoader,
	}
}

const (
	FlagApiServer       = "server"
	FlagMatchApiVersion = "match_server_version"
	FlagApiVersion      = "api_version"
	FlagAuthPath        = "auth_path"
	FlagInsecure        = "insecure_skip_tls_verify"
	FlagCertFile        = "client_certificate"
	FlagKeyFile         = "client_key"
	FlagCAFile          = "certificate_authority"
	FlagBearerToken     = "token"
)

// BindFlags implements Builder
func (builder *builder) BindFlags(flags *pflag.FlagSet) {
	flags.StringVarP(&builder.apiserver, FlagApiServer, "s", builder.apiserver, "The address of the Kubernetes API server")
	flags.BoolVar(&builder.matchApiVersion, FlagMatchApiVersion, false, "Require server version to match client version")
	flags.StringVar(&builder.apiVersion, FlagApiVersion, latest.Version, "The API version to use when talking to the server")
	flags.StringVarP(&builder.authPath, FlagAuthPath, "a", os.Getenv("HOME")+"/.kubernetes_auth", "Path to the auth info file. If missing, prompt the user. Only used if using https.")
	flags.Var(&builder.cmdAuthInfo.Insecure, FlagInsecure, "If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.")
	flags.Var(&builder.cmdAuthInfo.CertFile, FlagCertFile, "Path to a client key file for TLS.")
	flags.Var(&builder.cmdAuthInfo.KeyFile, FlagKeyFile, "Path to a client key file for TLS.")
	flags.Var(&builder.cmdAuthInfo.CAFile, FlagCAFile, "Path to a cert. file for the certificate authority.")
	flags.Var(&builder.cmdAuthInfo.BearerToken, FlagBearerToken, "Bearer token for authentication to the API server.")
}

// Client implements Builder
func (builder *builder) Client() (*client.Client, error) {
	clientConfig, err := builder.Config()
	if err != nil {
		return nil, err
	}

	c, err := client.New(clientConfig)
	if err != nil {
		return nil, err
	}

	if builder.matchApiVersion {
		clientVersion := version.Get()
		serverVersion, err := c.ServerVersion()
		if err != nil {
			return nil, fmt.Errorf("couldn't read version from server: %v\n", err)
		}
		if s := *serverVersion; !reflect.DeepEqual(clientVersion, s) {
			return nil, fmt.Errorf("server version (%#v) differs from client version (%#v)!\n", s, clientVersion)
		}
	}

	return c, nil
}

// Config implements Builder
func (builder *builder) Config() (*client.Config, error) {
	clientConfig := client.Config{}
	if len(builder.apiserver) > 0 {
		clientConfig.Host = builder.apiserver
	} else if len(os.Getenv("KUBERNETES_MASTER")) > 0 {
		clientConfig.Host = os.Getenv("KUBERNETES_MASTER")
	} else {
		// TODO: eventually apiserver should start on 443 and be secure by default
		clientConfig.Host = "http://localhost:8080"
	}
	clientConfig.Version = builder.apiVersion

	// only try to read the auth information if we are secure
	if client.IsConfigTransportTLS(&clientConfig) {
		authInfoFileFound := true
		authInfo, err := builder.authLoader.LoadAuth(builder.authPath)
		if authInfo == nil && err != nil { // only consider failing if we don't have any auth info
			if os.IsNotExist(err) { // if it's just a case of a missing file, simply flag the auth as not found and use the command line arguments
				authInfoFileFound = false
				authInfo = &clientauth.Info{}
			} else {
				return nil, err
			}
		}

		// If provided, the command line options override options from the auth file
		if !authInfoFileFound || builder.cmdAuthInfo.User.Provided() {
			authInfo.User = builder.cmdAuthInfo.User.Value
		}
		if !authInfoFileFound || builder.cmdAuthInfo.Password.Provided() {
			authInfo.Password = builder.cmdAuthInfo.Password.Value
		}
		if !authInfoFileFound || builder.cmdAuthInfo.CAFile.Provided() {
			authInfo.CAFile = builder.cmdAuthInfo.CAFile.Value
		}
		if !authInfoFileFound || builder.cmdAuthInfo.CertFile.Provided() {
			authInfo.CertFile = builder.cmdAuthInfo.CertFile.Value
		}
		if !authInfoFileFound || builder.cmdAuthInfo.KeyFile.Provided() {
			authInfo.KeyFile = builder.cmdAuthInfo.KeyFile.Value
		}
		if !authInfoFileFound || builder.cmdAuthInfo.BearerToken.Provided() {
			authInfo.BearerToken = builder.cmdAuthInfo.BearerToken.Value
		}
		if !authInfoFileFound || builder.cmdAuthInfo.Insecure.Provided() {
			authInfo.Insecure = &builder.cmdAuthInfo.Insecure.Value
		}

		clientConfig, err = authInfo.MergeWithConfig(clientConfig)
		if err != nil {
			return nil, err
		}
	}

	return &clientConfig, nil
}

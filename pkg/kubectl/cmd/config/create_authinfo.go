/*
Copyright 2014 The Kubernetes Authors.

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

package config

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"path/filepath"
	"strings"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/flag"
)

type createAuthInfoOptions struct {
	configAccess      clientcmd.ConfigAccess
	name              string
	authPath          util.StringFlag
	clientCertificate util.StringFlag
	clientKey         util.StringFlag
	token             util.StringFlag
	username          util.StringFlag
	password          util.StringFlag
	embedCertData     flag.Tristate
	authProvider      util.StringFlag

	authProviderArgs         map[string]string
	authProviderArgsToRemove []string
}

const (
	flagAuthProvider    = "auth-provider"
	flagAuthProviderArg = "auth-provider-arg"
)

var (
	create_authinfo_long = fmt.Sprintf(`
Sets a user entry in kubeconfig
Specifying a name that already exists will merge new fields on top of existing values.

  Client-certificate flags:
    --%v=certfile --%v=keyfile

  Bearer token flags:
    --%v=bearer_token

  Basic auth flags:
    --%v=basic_user --%v=basic_password

  Bearer token and basic auth are mutually exclusive.
`, clientcmd.FlagCertFile, clientcmd.FlagKeyFile, clientcmd.FlagBearerToken, clientcmd.FlagUsername, clientcmd.FlagPassword)

	create_authinfo_example = dedent.Dedent(`
		# Set only the "client-key" field on the "cluster-admin"
		# entry, without touching other values:
		kubectl config set-credentials cluster-admin --client-key=~/.kube/admin.key

		# Set basic auth for the "cluster-admin" entry
		kubectl config set-credentials cluster-admin --username=admin --password=uXFGweU9l35qcif

		# Embed client certificate data in the "cluster-admin" entry
		kubectl config set-credentials cluster-admin --client-certificate=~/.kube/admin.crt --embed-certs=true

		# Enable the Google Compute Platform auth provider for the "cluster-admin" entry
		kubectl config set-credentials cluster-admin --auth-provider=gcp

		# Enable the OpenID Connect auth provider for the "cluster-admin" entry with additional args
		kubectl config set-credentials cluster-admin --auth-provider=oidc --auth-provider-arg=client-id=foo --auth-provider-arg=client-secret=bar

		# Remove the "client-secret" config value for the OpenID Connect auth provider for the "cluster-admin" entry
		kubectl config set-credentials cluster-admin --auth-provider=oidc --auth-provider-arg=client-secret-`)
)

func NewCmdConfigSetAuthInfo(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &createAuthInfoOptions{configAccess: configAccess}
	return newCmdConfigSetAuthInfo(out, options)
}

func newCmdConfigSetAuthInfo(out io.Writer, options *createAuthInfoOptions) *cobra.Command {
	cmd := &cobra.Command{
		Use:     fmt.Sprintf("set-credentials NAME [--%v=path/to/certfile] [--%v=path/to/keyfile] [--%v=bearer_token] [--%v=basic_user] [--%v=basic_password] [--%v=provider_name] [--%v=key=value]", clientcmd.FlagCertFile, clientcmd.FlagKeyFile, clientcmd.FlagBearerToken, clientcmd.FlagUsername, clientcmd.FlagPassword, flagAuthProvider, flagAuthProviderArg),
		Short:   "Sets a user entry in kubeconfig",
		Long:    create_authinfo_long,
		Example: create_authinfo_example,
		Run: func(cmd *cobra.Command, args []string) {
			if !options.complete(cmd, out) {
				cmd.Help()
				return
			}

			err := options.run()
			if err != nil {
				fmt.Fprintf(out, "%v\n", err)
			} else {
				fmt.Fprintf(out, "user %q set.\n", options.name)
			}
		},
	}

	cmd.Flags().Var(&options.clientCertificate, clientcmd.FlagCertFile, "path to "+clientcmd.FlagCertFile+" file for the user entry in kubeconfig")
	cmd.MarkFlagFilename(clientcmd.FlagCertFile)
	cmd.Flags().Var(&options.clientKey, clientcmd.FlagKeyFile, "path to "+clientcmd.FlagKeyFile+" file for the user entry in kubeconfig")
	cmd.MarkFlagFilename(clientcmd.FlagKeyFile)
	cmd.Flags().Var(&options.token, clientcmd.FlagBearerToken, clientcmd.FlagBearerToken+" for the user entry in kubeconfig")
	cmd.Flags().Var(&options.username, clientcmd.FlagUsername, clientcmd.FlagUsername+" for the user entry in kubeconfig")
	cmd.Flags().Var(&options.password, clientcmd.FlagPassword, clientcmd.FlagPassword+" for the user entry in kubeconfig")
	cmd.Flags().Var(&options.authProvider, flagAuthProvider, "auth provider for the user entry in kubeconfig")
	cmd.Flags().StringSlice(flagAuthProviderArg, nil, "'key=value' arugments for the auth provider")
	f := cmd.Flags().VarPF(&options.embedCertData, clientcmd.FlagEmbedCerts, "", "embed client cert/key for the user entry in kubeconfig")
	f.NoOptDefVal = "true"

	return cmd
}

func (o createAuthInfoOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	startingStanza, exists := config.AuthInfos[o.name]
	if !exists {
		startingStanza = clientcmdapi.NewAuthInfo()
	}
	authInfo := o.modifyAuthInfo(*startingStanza)
	config.AuthInfos[o.name] = &authInfo

	if err := clientcmd.ModifyConfig(o.configAccess, *config, true); err != nil {
		return err
	}

	return nil
}

// authInfo builds an AuthInfo object from the options
func (o *createAuthInfoOptions) modifyAuthInfo(existingAuthInfo clientcmdapi.AuthInfo) clientcmdapi.AuthInfo {
	modifiedAuthInfo := existingAuthInfo

	var setToken, setBasic bool

	if o.clientCertificate.Provided() {
		certPath := o.clientCertificate.Value()
		if o.embedCertData.Value() {
			modifiedAuthInfo.ClientCertificateData, _ = ioutil.ReadFile(certPath)
			modifiedAuthInfo.ClientCertificate = ""
		} else {
			certPath, _ = filepath.Abs(certPath)
			modifiedAuthInfo.ClientCertificate = certPath
			if len(modifiedAuthInfo.ClientCertificate) > 0 {
				modifiedAuthInfo.ClientCertificateData = nil
			}
		}
	}
	if o.clientKey.Provided() {
		keyPath := o.clientKey.Value()
		if o.embedCertData.Value() {
			modifiedAuthInfo.ClientKeyData, _ = ioutil.ReadFile(keyPath)
			modifiedAuthInfo.ClientKey = ""
		} else {
			keyPath, _ = filepath.Abs(keyPath)
			modifiedAuthInfo.ClientKey = keyPath
			if len(modifiedAuthInfo.ClientKey) > 0 {
				modifiedAuthInfo.ClientKeyData = nil
			}
		}
	}

	if o.token.Provided() {
		modifiedAuthInfo.Token = o.token.Value()
		setToken = len(modifiedAuthInfo.Token) > 0
	}

	if o.username.Provided() {
		modifiedAuthInfo.Username = o.username.Value()
		setBasic = setBasic || len(modifiedAuthInfo.Username) > 0
	}
	if o.password.Provided() {
		modifiedAuthInfo.Password = o.password.Value()
		setBasic = setBasic || len(modifiedAuthInfo.Password) > 0
	}
	if o.authProvider.Provided() {
		newName := o.authProvider.Value()

		// Only overwrite if the existing auth-provider is nil, or different than the newly specified one.
		if modifiedAuthInfo.AuthProvider == nil || modifiedAuthInfo.AuthProvider.Name != newName {
			modifiedAuthInfo.AuthProvider = &clientcmdapi.AuthProviderConfig{
				Name: newName,
			}
		}
	}

	if modifiedAuthInfo.AuthProvider != nil {
		if modifiedAuthInfo.AuthProvider.Config == nil {
			modifiedAuthInfo.AuthProvider.Config = make(map[string]string)
		}
		for _, toRemove := range o.authProviderArgsToRemove {
			delete(modifiedAuthInfo.AuthProvider.Config, toRemove)
		}
		for key, value := range o.authProviderArgs {
			modifiedAuthInfo.AuthProvider.Config[key] = value
		}
	}

	// If any auth info was set, make sure any other existing auth types are cleared
	if setToken || setBasic {
		if !setToken {
			modifiedAuthInfo.Token = ""
		}
		if !setBasic {
			modifiedAuthInfo.Username = ""
			modifiedAuthInfo.Password = ""
		}
	}

	return modifiedAuthInfo
}

func (o *createAuthInfoOptions) complete(cmd *cobra.Command, out io.Writer) bool {
	args := cmd.Flags().Args()
	if len(args) != 1 {
		return false
	}

	authProviderArgs, err := cmd.Flags().GetStringSlice(flagAuthProviderArg)
	if err != nil {
		fmt.Fprintf(out, "Error: %s\n", err)
		return false
	}

	if len(authProviderArgs) > 0 {
		newPairs, removePairs, err := cmdutil.ParsePairs(authProviderArgs, flagAuthProviderArg, true)
		if err != nil {
			fmt.Fprintf(out, "Error: %s\n", err)
			return false
		}
		o.authProviderArgs = newPairs
		o.authProviderArgsToRemove = removePairs
	}

	o.name = args[0]
	return true
}

func (o createAuthInfoOptions) validate() error {
	if len(o.name) == 0 {
		return errors.New("you must specify a non-empty user name")
	}
	methods := []string{}
	if len(o.token.Value()) > 0 {
		methods = append(methods, fmt.Sprintf("--%v", clientcmd.FlagBearerToken))
	}
	if len(o.username.Value()) > 0 || len(o.password.Value()) > 0 {
		methods = append(methods, fmt.Sprintf("--%v/--%v", clientcmd.FlagUsername, clientcmd.FlagPassword))
	}
	if len(methods) > 1 {
		return fmt.Errorf("you cannot specify more than one authentication method at the same time: %v", strings.Join(methods, ", "))
	}
	if o.embedCertData.Value() {
		certPath := o.clientCertificate.Value()
		keyPath := o.clientKey.Value()
		if certPath == "" && keyPath == "" {
			return fmt.Errorf("you must specify a --%s or --%s to embed", clientcmd.FlagCertFile, clientcmd.FlagKeyFile)
		}
		if certPath != "" {
			if _, err := ioutil.ReadFile(certPath); err != nil {
				return fmt.Errorf("error reading %s data from %s: %v", clientcmd.FlagCertFile, certPath, err)
			}
		}
		if keyPath != "" {
			if _, err := ioutil.ReadFile(keyPath); err != nil {
				return fmt.Errorf("error reading %s data from %s: %v", clientcmd.FlagKeyFile, keyPath, err)
			}
		}
	}

	return nil
}

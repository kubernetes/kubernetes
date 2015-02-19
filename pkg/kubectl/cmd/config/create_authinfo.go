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

package config

import (
	"errors"
	"fmt"
	"io"
	"strings"

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

type createAuthInfoOptions struct {
	pathOptions       *pathOptions
	name              string
	authPath          util.StringFlag
	clientCertificate util.StringFlag
	clientKey         util.StringFlag
	token             util.StringFlag
	username          util.StringFlag
	password          util.StringFlag
}

func NewCmdConfigSetAuthInfo(out io.Writer, pathOptions *pathOptions) *cobra.Command {
	options := &createAuthInfoOptions{pathOptions: pathOptions}

	cmd := &cobra.Command{
		Use:   fmt.Sprintf("set-credentials name [--%v=authfile] [--%v=certfile] [--%v=keyfile] [--%v=bearer_token] [--%v=basic_user] [--%v=basic_password]", clientcmd.FlagAuthPath, clientcmd.FlagCertFile, clientcmd.FlagKeyFile, clientcmd.FlagBearerToken, clientcmd.FlagUsername, clientcmd.FlagPassword),
		Short: "Sets a user entry in .kubeconfig",
		Long: fmt.Sprintf(`Sets a user entry in .kubeconfig

  Specifying a name that already exists will merge new fields on top of existing
  values. For example, the following only sets the "client-key" field on the 
  "cluster-admin" entry, without touching other values:

    set-credentials cluster-admin --client-key=~/.kube/admin.key

  Client-certificate flags:
    --%v=certfile --%v=keyfile

  Bearer token flags:
    --%v=bearer_token

  Basic auth flags:
    --%v=basic_user --%v=basic_password

  Bearer token and basic auth are mutually exclusive.
`, clientcmd.FlagCertFile, clientcmd.FlagKeyFile, clientcmd.FlagBearerToken, clientcmd.FlagUsername, clientcmd.FlagPassword),
		Run: func(cmd *cobra.Command, args []string) {
			if !options.complete(cmd) {
				return
			}

			err := options.run()
			if err != nil {
				fmt.Fprintf(out, "%v\n", err)
			}
		},
	}

	cmd.Flags().Var(&options.authPath, clientcmd.FlagAuthPath, clientcmd.FlagAuthPath+" for the user entry in .kubeconfig")
	cmd.Flags().Var(&options.clientCertificate, clientcmd.FlagCertFile, clientcmd.FlagCertFile+" for the user entry in .kubeconfig")
	cmd.Flags().Var(&options.clientKey, clientcmd.FlagKeyFile, clientcmd.FlagKeyFile+" for the user entry in .kubeconfig")
	cmd.Flags().Var(&options.token, clientcmd.FlagBearerToken, clientcmd.FlagBearerToken+" for the user entry in .kubeconfig")
	cmd.Flags().Var(&options.username, clientcmd.FlagUsername, clientcmd.FlagUsername+" for the user entry in .kubeconfig")
	cmd.Flags().Var(&options.password, clientcmd.FlagPassword, clientcmd.FlagPassword+" for the user entry in .kubeconfig")

	return cmd
}

func (o createAuthInfoOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, filename, err := o.pathOptions.getStartingConfig()
	if err != nil {
		return err
	}

	authInfo := o.modifyAuthInfo(config.AuthInfos[o.name])
	config.AuthInfos[o.name] = authInfo

	err = clientcmd.WriteToFile(*config, filename)
	if err != nil {
		return err
	}

	return nil
}

// authInfo builds an AuthInfo object from the options
func (o *createAuthInfoOptions) modifyAuthInfo(existingAuthInfo clientcmdapi.AuthInfo) clientcmdapi.AuthInfo {
	modifiedAuthInfo := existingAuthInfo

	var setToken, setBasic bool

	if o.authPath.Provided() {
		modifiedAuthInfo.AuthPath = o.authPath.Value()
	}

	if o.clientCertificate.Provided() {
		modifiedAuthInfo.ClientCertificate = o.clientCertificate.Value()
		if len(modifiedAuthInfo.ClientCertificate) > 0 {
			modifiedAuthInfo.ClientCertificateData = nil
		}
	}
	if o.clientKey.Provided() {
		modifiedAuthInfo.ClientKey = o.clientKey.Value()
		if len(modifiedAuthInfo.ClientKey) > 0 {
			modifiedAuthInfo.ClientKeyData = nil
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

func (o *createAuthInfoOptions) complete(cmd *cobra.Command) bool {
	args := cmd.Flags().Args()
	if len(args) != 1 {
		cmd.Help()
		return false
	}

	o.name = args[0]
	return true
}

func (o createAuthInfoOptions) validate() error {
	if len(o.name) == 0 {
		return errors.New("You must specify a non-empty user name")
	}
	methods := []string{}
	if len(o.token.Value()) > 0 {
		methods = append(methods, fmt.Sprintf("--%v", clientcmd.FlagBearerToken))
	}
	if len(o.username.Value()) > 0 || len(o.password.Value()) > 0 {
		methods = append(methods, fmt.Sprintf("--%v/--%v", clientcmd.FlagUsername, clientcmd.FlagPassword))
	}
	if len(methods) > 1 {
		return fmt.Errorf("You cannot specify more than one authentication method at the same time: %v", strings.Join(methods, ", "))
	}

	return nil
}

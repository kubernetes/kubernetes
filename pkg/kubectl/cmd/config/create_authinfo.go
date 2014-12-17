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

	"github.com/spf13/cobra"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	clientcmdapi "github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd/api"
)

type createAuthInfoOptions struct {
	pathOptions       *pathOptions
	name              string
	authPath          string
	clientCertificate string
	clientKey         string
	token             string
}

func NewCmdConfigSetAuthInfo(out io.Writer, pathOptions *pathOptions) *cobra.Command {
	options := &createAuthInfoOptions{pathOptions: pathOptions}

	cmd := &cobra.Command{
		Use:   "set-credentials name",
		Short: "Sets a user entry in .kubeconfig",
		Long: `Sets a user entry in .kubeconfig

		Specifying a name that already exists overwrites that user entry.
		`,
		Run: func(cmd *cobra.Command, args []string) {
			if !options.complete(cmd) {
				return
			}

			err := options.run()
			if err != nil {
				fmt.Printf("%v\n", err)
			}
		},
	}

	cmd.Flags().StringVar(&options.authPath, clientcmd.FlagAuthPath, "", clientcmd.FlagAuthPath+" for the user entry in .kubeconfig")
	cmd.Flags().StringVar(&options.clientCertificate, clientcmd.FlagCertFile, "", clientcmd.FlagCertFile+" for the user entry in .kubeconfig")
	cmd.Flags().StringVar(&options.clientKey, clientcmd.FlagKeyFile, "", clientcmd.FlagKeyFile+" for the user entry in .kubeconfig")
	cmd.Flags().StringVar(&options.token, clientcmd.FlagBearerToken, "", clientcmd.FlagBearerToken+" for the user entry in .kubeconfig")

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

	authInfo := o.authInfo()
	config.AuthInfos[o.name] = authInfo

	err = clientcmd.WriteToFile(*config, filename)
	if err != nil {
		return err
	}

	return nil
}

// authInfo builds an AuthInfo object from the options
func (o *createAuthInfoOptions) authInfo() clientcmdapi.AuthInfo {
	authInfo := clientcmdapi.AuthInfo{
		AuthPath:          o.authPath,
		ClientCertificate: o.clientCertificate,
		ClientKey:         o.clientKey,
		Token:             o.token,
	}

	return authInfo
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

	return nil
}

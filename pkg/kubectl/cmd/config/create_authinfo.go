/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/client/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/clientcmd/api"
)

type createAuthInfoOptions struct {
	configAccess  ConfigAccess
	name          string
	certPath      string
	keyPath       string
	token         string
	username      string
	password      string
	embedCertData bool
}

var create_authinfo_long = fmt.Sprintf(`Sets a user entry in kubeconfig
Specifying a name that already exists will merge new fields on top of existing values.

  Client-certificate flags:
    --%v=certfile --%v=keyfile

  Bearer token flags:
    --%v=bearer_token

  Basic auth flags:
    --%v=basic_user --%v=basic_password

  Bearer token and basic auth are mutually exclusive.
`, clientcmd.FlagCertFile, clientcmd.FlagKeyFile, clientcmd.FlagBearerToken, clientcmd.FlagUsername, clientcmd.FlagPassword)

const create_authinfo_example = `// Set only the "client-key" field on the "cluster-admin"
// entry, without touching other values:
$ kubectl config set-credentials cluster-admin --client-key=~/.kube/admin.key

// Set basic auth for the "cluster-admin" entry
$ kubectl config set-credentials cluster-admin --username=admin --password=uXFGweU9l35qcif

// Embed client certificate data in the "cluster-admin" entry
$ kubectl config set-credentials cluster-admin --client-certificate=~/.kube/admin.crt --embed-certs=true`

func NewCmdConfigSetAuthInfo(out io.Writer, configAccess ConfigAccess) *cobra.Command {
	options := &createAuthInfoOptions{configAccess: configAccess}

	cmd := &cobra.Command{
		Use:     fmt.Sprintf("set-credentials NAME [--%v=path/to/certfile] [--%v=path/to/keyfile] [--%v=bearer_token] [--%v=basic_user] [--%v=basic_password]", clientcmd.FlagCertFile, clientcmd.FlagKeyFile, clientcmd.FlagBearerToken, clientcmd.FlagUsername, clientcmd.FlagPassword),
		Short:   "Sets a user entry in kubeconfig",
		Long:    create_authinfo_long,
		Example: create_authinfo_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := options.run(cmd, args)
			if err != nil {
				fmt.Fprintf(out, "%v\n", err)
			}
		},
	}

	cmd.Flags().StringVar(&options.certPath, clientcmd.FlagCertFile, "", "path to "+clientcmd.FlagCertFile+" for the user entry in kubeconfig")
	cmd.Flags().StringVar(&options.keyPath, clientcmd.FlagKeyFile, "", "path to "+clientcmd.FlagKeyFile+" for the user entry in kubeconfig")
	cmd.Flags().StringVar(&options.token, clientcmd.FlagBearerToken, "", clientcmd.FlagBearerToken+" for the user entry in kubeconfig")
	cmd.Flags().StringVar(&options.username, clientcmd.FlagUsername, "", clientcmd.FlagUsername+" for the user entry in kubeconfig")
	cmd.Flags().StringVar(&options.password, clientcmd.FlagPassword, "", clientcmd.FlagPassword+" for the user entry in kubeconfig")
	cmd.Flags().BoolVar(&options.embedCertData, clientcmd.FlagEmbedCerts, false, "embed client cert/key for the user entry in kubeconfig")

	return cmd
}

func (o createAuthInfoOptions) run(cmd *cobra.Command, args []string) error {
	if len(args) != 1 {
		cmd.Help()
		return nil
	}

	o.name = args[0]

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
	authInfo := o.modifyAuthInfo(cmd, *startingStanza)
	config.AuthInfos[o.name] = &authInfo

	if err := ModifyConfig(o.configAccess, *config, true); err != nil {
		return err
	}

	return nil
}

// authInfo builds an AuthInfo object from the options
func (o *createAuthInfoOptions) modifyAuthInfo(cmd *cobra.Command, existingAuthInfo clientcmdapi.AuthInfo) clientcmdapi.AuthInfo {
	modifiedAuthInfo := existingAuthInfo

	var setToken, setBasic bool

	if cmd.Flags().Changed(clientcmd.FlagCertFile) {
		certPath := o.certPath
		if o.embedCertData {
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
	if cmd.Flags().Changed(clientcmd.FlagKeyFile) {
		keyPath := o.keyPath
		if o.embedCertData {
			modifiedAuthInfo.ClientKeyData, _ = ioutil.ReadFile(keyPath)
			modifiedAuthInfo.ClientKey = ""
		} else {
			keyPath, _ := filepath.Abs(keyPath)
			modifiedAuthInfo.ClientKey = keyPath
			if len(modifiedAuthInfo.ClientKey) > 0 {
				modifiedAuthInfo.ClientKeyData = nil
			}
		}
	}

	if cmd.Flags().Changed(clientcmd.FlagBearerToken) {
		modifiedAuthInfo.Token = o.token
		setToken = len(modifiedAuthInfo.Token) > 0
	}

	if cmd.Flags().Changed(clientcmd.FlagUsername) {
		modifiedAuthInfo.Username = o.username
		setBasic = setBasic || len(modifiedAuthInfo.Username) > 0
	}

	if cmd.Flags().Changed(clientcmd.FlagPassword) {
		modifiedAuthInfo.Password = o.password
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

func (o createAuthInfoOptions) validate() error {
	if len(o.name) == 0 {
		return errors.New("You must specify a non-empty user name")
	}
	methods := []string{}
	if o.token != "" {
		methods = append(methods, fmt.Sprintf("--%v", clientcmd.FlagBearerToken))
	}
	if o.username != "" || o.password != "" {
		methods = append(methods, fmt.Sprintf("--%v/--%v", clientcmd.FlagUsername, clientcmd.FlagPassword))
	}
	if len(methods) > 1 {
		return fmt.Errorf("You cannot specify more than one authentication method at the same time: %v", strings.Join(methods, ", "))
	}
	if o.embedCertData {
		certPath := o.certPath
		keyPath := o.keyPath
		if certPath == "" && keyPath == "" {
			return fmt.Errorf("You must specify a --%s or --%s to embed", clientcmd.FlagCertFile, clientcmd.FlagKeyFile)
		}
		if certPath != "" {
			if _, err := ioutil.ReadFile(certPath); err != nil {
				return fmt.Errorf("Error reading %s data from %s: %v", clientcmd.FlagCertFile, certPath, err)
			}
		}
		if keyPath != "" {
			if _, err := ioutil.ReadFile(keyPath); err != nil {
				return fmt.Errorf("Error reading %s data from %s: %v", clientcmd.FlagKeyFile, keyPath, err)
			}
		}
	}

	return nil
}

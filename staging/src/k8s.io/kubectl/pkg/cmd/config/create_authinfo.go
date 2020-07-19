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
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cliflag "k8s.io/component-base/cli/flag"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

type createAuthInfoOptions struct {
	configAccess      clientcmd.ConfigAccess
	name              string
	clientCertificate cliflag.StringFlag
	clientKey         cliflag.StringFlag
	token             cliflag.StringFlag
	username          cliflag.StringFlag
	password          cliflag.StringFlag
	embedCertData     cliflag.Tristate
	authProvider      cliflag.StringFlag

	authProviderArgs         map[string]string
	authProviderArgsToRemove []string

	execCommand     cliflag.StringFlag
	execAPIVersion  cliflag.StringFlag
	execArgs        []string
	execEnv         map[string]string
	execEnvToRemove []string
}

const (
	flagAuthProvider    = "auth-provider"
	flagAuthProviderArg = "auth-provider-arg"

	flagExecCommand    = "exec-command"
	flagExecAPIVersion = "exec-api-version"
	flagExecArg        = "exec-arg"
	flagExecEnv        = "exec-env"
)

var (
	createAuthInfoLong = fmt.Sprintf(templates.LongDesc(`
		Sets a user entry in kubeconfig

		Specifying a name that already exists will merge new fields on top of existing values.

		    Client-certificate flags:
		    --%v=certfile --%v=keyfile

		    Bearer token flags:
			  --%v=bearer_token

		    Basic auth flags:
			  --%v=basic_user --%v=basic_password

		Bearer token and basic auth are mutually exclusive.`), clientcmd.FlagCertFile, clientcmd.FlagKeyFile, clientcmd.FlagBearerToken, clientcmd.FlagUsername, clientcmd.FlagPassword)

	createAuthInfoExample = templates.Examples(`
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
		kubectl config set-credentials cluster-admin --auth-provider=oidc --auth-provider-arg=client-secret-

		# Enable new exec auth plugin for the "cluster-admin" entry
		kubectl config set-credentials cluster-admin --exec-command=/path/to/the/executable --exec-api-version=client.authentication.k8s.io/v1beta1

		# Define new exec auth plugin args for the "cluster-admin" entry
		kubectl config set-credentials cluster-admin --exec-arg=arg1 --exec-arg=arg2

		# Create or update exec auth plugin environment variables for the "cluster-admin" entry
		kubectl config set-credentials cluster-admin --exec-env=key1=val1 --exec-env=key2=val2

		# Remove exec auth plugin environment variables for the "cluster-admin" entry
		kubectl config set-credentials cluster-admin --exec-env=var-to-remove-`)
)

// NewCmdConfigSetAuthInfo returns an Command option instance for 'config set-credentials' sub command
func NewCmdConfigSetAuthInfo(out io.Writer, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &createAuthInfoOptions{configAccess: configAccess}
	return newCmdConfigSetAuthInfo(out, options)
}

func newCmdConfigSetAuthInfo(out io.Writer, options *createAuthInfoOptions) *cobra.Command {
	cmd := &cobra.Command{
		Use: fmt.Sprintf(
			"set-credentials NAME [--%v=path/to/certfile] "+
				"[--%v=path/to/keyfile] "+
				"[--%v=bearer_token] "+
				"[--%v=basic_user] "+
				"[--%v=basic_password] "+
				"[--%v=provider_name] "+
				"[--%v=key=value] "+
				"[--%v=exec_command] "+
				"[--%v=exec_api_version] "+
				"[--%v=arg] "+
				"[--%v=key=value]",
			clientcmd.FlagCertFile,
			clientcmd.FlagKeyFile,
			clientcmd.FlagBearerToken,
			clientcmd.FlagUsername,
			clientcmd.FlagPassword,
			flagAuthProvider,
			flagAuthProviderArg,
			flagExecCommand,
			flagExecAPIVersion,
			flagExecArg,
			flagExecEnv,
		),
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Sets a user entry in kubeconfig"),
		Long:                  createAuthInfoLong,
		Example:               createAuthInfoExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := options.complete(cmd, out)
			if err != nil {
				cmd.Help()
				cmdutil.CheckErr(err)
			}
			cmdutil.CheckErr(options.run())
			fmt.Fprintf(out, "User %q set.\n", options.name)
		},
	}

	cmd.Flags().Var(&options.clientCertificate, clientcmd.FlagCertFile, "Path to "+clientcmd.FlagCertFile+" file for the user entry in kubeconfig")
	cmd.MarkFlagFilename(clientcmd.FlagCertFile)
	cmd.Flags().Var(&options.clientKey, clientcmd.FlagKeyFile, "Path to "+clientcmd.FlagKeyFile+" file for the user entry in kubeconfig")
	cmd.MarkFlagFilename(clientcmd.FlagKeyFile)
	cmd.Flags().Var(&options.token, clientcmd.FlagBearerToken, clientcmd.FlagBearerToken+" for the user entry in kubeconfig")
	cmd.Flags().Var(&options.username, clientcmd.FlagUsername, clientcmd.FlagUsername+" for the user entry in kubeconfig")
	cmd.Flags().Var(&options.password, clientcmd.FlagPassword, clientcmd.FlagPassword+" for the user entry in kubeconfig")
	cmd.Flags().Var(&options.authProvider, flagAuthProvider, "Auth provider for the user entry in kubeconfig")
	cmd.Flags().StringSlice(flagAuthProviderArg, nil, "'key=value' arguments for the auth provider")
	cmd.Flags().Var(&options.execCommand, flagExecCommand, "Command for the exec credential plugin for the user entry in kubeconfig")
	cmd.Flags().Var(&options.execAPIVersion, flagExecAPIVersion, "API version of the exec credential plugin for the user entry in kubeconfig")
	cmd.Flags().StringSlice(flagExecArg, nil, "New arguments for the exec credential plugin command for the user entry in kubeconfig")
	cmd.Flags().StringArray(flagExecEnv, nil, "'key=value' environment values for the exec credential plugin")
	f := cmd.Flags().VarPF(&options.embedCertData, clientcmd.FlagEmbedCerts, "", "Embed client cert/key for the user entry in kubeconfig")
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

	if o.execCommand.Provided() {
		newExecCommand := o.execCommand.Value()

		// create new Exec if doesn't exist, otherwise just modify the command
		if modifiedAuthInfo.Exec == nil {
			modifiedAuthInfo.Exec = &clientcmdapi.ExecConfig{
				Command: newExecCommand,
			}
		} else {
			modifiedAuthInfo.Exec.Command = newExecCommand
			// explicitly reset exec arguments
			modifiedAuthInfo.Exec.Args = nil
		}
	}

	// modify next values only if Exec exists, ignore these changes otherwise
	if modifiedAuthInfo.Exec != nil {
		if o.execAPIVersion.Provided() {
			modifiedAuthInfo.Exec.APIVersion = o.execAPIVersion.Value()
		}

		// rewrite exec arguments list with new values
		if o.execArgs != nil {
			modifiedAuthInfo.Exec.Args = o.execArgs
		}

		// iterate over the existing exec env values and remove the specified
		if o.execEnvToRemove != nil {
			newExecEnv := []clientcmdapi.ExecEnvVar{}
			for _, value := range modifiedAuthInfo.Exec.Env {
				needToRemove := false
				for _, elemToRemove := range o.execEnvToRemove {
					if value.Name == elemToRemove {
						needToRemove = true
						break
					}
				}
				if !needToRemove {
					newExecEnv = append(newExecEnv, value)
				}
			}
			modifiedAuthInfo.Exec.Env = newExecEnv
		}

		// update or create specified environment variables for the exec plugin
		if o.execEnv != nil {
			newEnv := []clientcmdapi.ExecEnvVar{}
			for newEnvName, newEnvValue := range o.execEnv {
				needToCreate := true
				for i := 0; i < len(modifiedAuthInfo.Exec.Env); i++ {
					if modifiedAuthInfo.Exec.Env[i].Name == newEnvName {
						// update the existing value
						needToCreate = false
						modifiedAuthInfo.Exec.Env[i].Value = newEnvValue
						break
					}
				}
				if needToCreate {
					// create a new env value
					newEnv = append(newEnv, clientcmdapi.ExecEnvVar{Name: newEnvName, Value: newEnvValue})
				}
			}
			modifiedAuthInfo.Exec.Env = append(modifiedAuthInfo.Exec.Env, newEnv...)
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

func (o *createAuthInfoOptions) complete(cmd *cobra.Command, out io.Writer) error {
	args := cmd.Flags().Args()
	if len(args) != 1 {
		return fmt.Errorf("Unexpected args: %v", args)
	}

	authProviderArgs, err := cmd.Flags().GetStringSlice(flagAuthProviderArg)
	if err != nil {
		return fmt.Errorf("Error: %s", err)
	}

	if len(authProviderArgs) > 0 {
		newPairs, removePairs, err := cmdutil.ParsePairs(authProviderArgs, flagAuthProviderArg, true)
		if err != nil {
			return fmt.Errorf("Error: %s", err)
		}
		o.authProviderArgs = newPairs
		o.authProviderArgsToRemove = removePairs
	}

	execArgs, err := cmd.Flags().GetStringSlice(flagExecArg)
	if err != nil {
		return fmt.Errorf("Error: %s", err)
	}
	if len(execArgs) > 0 {
		o.execArgs = execArgs
	}

	execEnv, err := cmd.Flags().GetStringArray(flagExecEnv)
	if err != nil {
		return fmt.Errorf("Error: %s", err)
	}
	if len(execEnv) > 0 {
		newPairs, removePairs, err := cmdutil.ParsePairs(execEnv, flagExecEnv, true)
		if err != nil {
			return fmt.Errorf("Error: %s", err)
		}
		o.execEnv = newPairs
		o.execEnvToRemove = removePairs
	}

	o.name = args[0]
	return nil
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
			if _, err := os.Stat(certPath); err != nil {
				return fmt.Errorf("could not stat %s file %s: %v", clientcmd.FlagCertFile, certPath, err)
			}
		}
		if keyPath != "" {
			if _, err := os.Stat(keyPath); err != nil {
				return fmt.Errorf("could not stat %s file %s: %v", clientcmd.FlagKeyFile, keyPath, err)
			}
		}
	}

	return nil
}

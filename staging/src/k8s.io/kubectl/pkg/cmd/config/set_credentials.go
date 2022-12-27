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
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cliflag "k8s.io/component-base/cli/flag"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

const (
	flagAuthProvider    = "auth-provider"
	flagAuthProviderArg = "auth-provider-arg"

	flagExecCommand    = "exec-command"
	flagExecAPIVersion = "exec-api-version"
	flagExecArg        = "exec-arg"
	flagExecEnv        = "exec-env"
)

var (
	setCredentialsLong = fmt.Sprintf(templates.LongDesc(i18n.T(`
		Set a user entry in kubeconfig.

		Specifying a name that already exists will merge new fields on top of existing values.

		    Client-certificate flags:
		    --%v=certfile --%v=keyfile

		    Bearer token flags:
			  --%v=bearer_token

		    Basic auth flags:
			  --%v=basic_user --%v=basic_password

		Bearer token and basic auth are mutually exclusive.`)),
		clientcmd.FlagCertFile,
		clientcmd.FlagKeyFile,
		clientcmd.FlagBearerToken,
		clientcmd.FlagUsername,
		clientcmd.FlagPassword,
	)

	setCredentialsExample = templates.Examples(`
		# Set only the "client-key" field on the "cluster-admin"
		# entry, without touching other values
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

type SetCredentialFlags struct {
	authProvider      cliflag.StringFlag
	authProviderArg   []string
	clientCertificate cliflag.StringFlag
	clientKey         cliflag.StringFlag
	embedCertData     bool
	execAPIVersion    cliflag.StringFlag
	execArg           []string
	execCommand       cliflag.StringFlag
	execEnv           []string
	token             cliflag.StringFlag
	password          cliflag.StringFlag
	username          cliflag.StringFlag

	configAccess clientcmd.ConfigAccess
}

type SetCredentialsOptions struct {
	authProvider             cliflag.StringFlag
	authProviderArgs         map[string]string
	authProviderArgsToRemove []string
	clientCertificate        cliflag.StringFlag
	clientKey                cliflag.StringFlag
	embedCertData            bool
	execAPIVersion           cliflag.StringFlag
	execArgs                 []string
	execCommand              cliflag.StringFlag
	execEnv                  map[string]string
	execEnvToRemove          []string
	name                     string
	password                 cliflag.StringFlag `datapolicy:"password"`
	token                    cliflag.StringFlag `datapolicy:"token"`
	username                 cliflag.StringFlag

	configAccess clientcmd.ConfigAccess
	ioStream     genericclioptions.IOStreams
}

func NewCmdConfigSetCredentials(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	flags := NewSetCredentialFlags(configAccess)

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
				"[--%v=key=value]"+
				"[--%v]",
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
			clientcmd.FlagEmbedCerts,
		),
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Set a user entry in kubeconfig"),
		Long:                  setCredentialsLong,
		Example:               setCredentialsExample,
		Run: func(cmd *cobra.Command, args []string) {
			options, err := flags.ToOptions(streams, args)
			cmdutil.CheckErr(err)
			cmdutil.CheckErr(options.RunSetCredentials())
		},
	}

	if err := flags.AddFlags(cmd); err != nil {
		cmdutil.CheckErr(err)
	}

	return cmd
}

func NewSetCredentialFlags(configAccess clientcmd.ConfigAccess) *SetCredentialFlags {
	return &SetCredentialFlags{
		authProvider:      cliflag.StringFlag{},
		authProviderArg:   []string{},
		clientCertificate: cliflag.StringFlag{},
		clientKey:         cliflag.StringFlag{},
		configAccess:      configAccess,
		embedCertData:     false,
		execAPIVersion:    cliflag.StringFlag{},
		execArg:           []string{},
		execCommand:       cliflag.StringFlag{},
		execEnv:           []string{},
		password:          cliflag.StringFlag{},
		token:             cliflag.StringFlag{},
		username:          cliflag.StringFlag{},
	}
}

// AddFlags registers flags for a cli
func (flags *SetCredentialFlags) AddFlags(cmd *cobra.Command) error {
	cmd.Flags().Var(&flags.authProvider, flagAuthProvider, "Auth provider for the user entry in kubeconfig")
	cmd.Flags().StringSliceVar(&flags.authProviderArg, flagAuthProviderArg, nil, "'key=value' arguments for the auth provider")

	cmd.Flags().Var(&flags.clientCertificate, clientcmd.FlagCertFile, "Path to "+clientcmd.FlagCertFile+" file for the user entry in kubeconfig")
	if err := cmd.MarkFlagFilename(clientcmd.FlagCertFile); err != nil {
		return err
	}

	cmd.Flags().Var(&flags.clientKey, clientcmd.FlagKeyFile, "Path to "+clientcmd.FlagKeyFile+" file for the user entry in kubeconfig")
	if err := cmd.MarkFlagFilename(clientcmd.FlagKeyFile); err != nil {
		return err
	}

	cmd.Flags().BoolVar(&flags.embedCertData, clientcmd.FlagEmbedCerts, false, "Embed client cert/key for the user entry in kubeconfig")
	cmd.Flags().Var(&flags.execAPIVersion, flagExecAPIVersion, "API version of the exec credential plugin for the user entry in kubeconfig")
	cmd.Flags().StringSliceVar(&flags.execArg, flagExecArg, nil, "New arguments for the exec credential plugin command for the user entry in kubeconfig")
	cmd.Flags().Var(&flags.execCommand, flagExecCommand, "Command for the exec credential plugin for the user entry in kubeconfig")
	cmd.Flags().StringArrayVar(&flags.execEnv, flagExecEnv, nil, "'key=value' environment values for the exec credential plugin")
	cmd.Flags().Var(&flags.password, clientcmd.FlagPassword, clientcmd.FlagPassword+" for the user entry in kubeconfig")
	cmd.Flags().Var(&flags.token, clientcmd.FlagBearerToken, clientcmd.FlagBearerToken+" for the user entry in kubeconfig")
	cmd.Flags().Var(&flags.username, clientcmd.FlagUsername, clientcmd.FlagUsername+" for the user entry in kubeconfig")
	return nil
}

// ToOptions converts from CLI inputs to runtime inputs
func (flags *SetCredentialFlags) ToOptions(streams genericclioptions.IOStreams, args []string) (*SetCredentialsOptions, error) {
	if len(args) < 1 {
		return nil, fmt.Errorf("you must specify a user")
	} else if len(args) > 1 {
		return nil, fmt.Errorf("you may only specify one user")
	}

	if flags.token.Provided() && (flags.username.Provided() || flags.password.Provided()) {
		return nil, fmt.Errorf("cannot provide both --%s and --%s or --%s at the same time", clientcmd.FlagBearerToken, clientcmd.FlagUsername, clientcmd.FlagPassword)
	}

	if flags.embedCertData && !(flags.clientCertificate.Provided() || flags.clientKey.Provided()) {
		return nil, fmt.Errorf("must provide --%s or --%s when --%s is set", clientcmd.FlagKeyFile, clientcmd.FlagCertFile, clientcmd.FlagEmbedCerts)
	}

	options := &SetCredentialsOptions{
		authProvider:      flags.authProvider,
		clientCertificate: flags.clientCertificate,
		clientKey:         flags.clientKey,
		configAccess:      flags.configAccess,
		embedCertData:     flags.embedCertData,
		execAPIVersion:    flags.execAPIVersion,
		execArgs:          flags.execArg,
		execCommand:       flags.execCommand,
		ioStream:          streams,
		name:              args[0],
		password:          flags.password,
		token:             flags.token,
		username:          flags.username,
	}

	if len(flags.authProviderArg) > 0 {
		lastElem := flags.authProviderArg[len(flags.authProviderArg)-1]
		if lastElem[len(lastElem)-1:] == "-" {
			options.authProviderArgsToRemove = flags.authProviderArg
		} else {
			authProviderArgs := make(map[string]string)
			for _, authProviderArg := range flags.authProviderArg {
				argList := strings.Split(authProviderArg, "=")
				if len(argList) != 2 {
					return nil, fmt.Errorf("auth provider arg flag must use the format --auth-provider-arg=key=value or --auth-provider-arg=key- ")
				}
				authProviderArgs[argList[0]] = argList[1]
			}
			options.authProviderArgs = authProviderArgs
		}
	}

	if len(flags.execEnv) > 0 {
		lastElem := flags.execEnv[len(flags.execEnv)-1]
		if lastElem[len(lastElem)-1:] == "-" {
			options.execEnvToRemove = flags.execEnv
		} else {
			execEnvs := make(map[string]string, len(flags.execEnv))
			for _, execEnv := range flags.execEnv {
				argList := strings.Split(execEnv, "=")
				if len(argList) != 2 {
					return nil, fmt.Errorf("auth provider arg flag must use the format --auth-provider-arg=key=value or --auth-provider-arg=key- ")
				}
				execEnvs[argList[0]] = argList[1]
			}
			options.authProviderArgs = execEnvs
		}
	}

	return options, nil
}

func (o *SetCredentialsOptions) RunSetCredentials() error {
	config, _, err := loadConfig(o.configAccess)
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

	if _, err := fmt.Fprintf(o.ioStream.Out, "User %q set.\n", o.name); err != nil {
		return err
	}

	return nil
}

func (o *SetCredentialsOptions) modifyAuthInfo(existingAuthInfo clientcmdapi.AuthInfo) clientcmdapi.AuthInfo {
	modifiedAuthInfo := existingAuthInfo

	var setToken, setBasic bool

	if o.clientCertificate.Provided() {
		certPath := o.clientCertificate.Value()
		if o.embedCertData {
			modifiedAuthInfo.ClientCertificateData, _ = os.ReadFile(certPath)
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
		if o.embedCertData {
			modifiedAuthInfo.ClientKeyData, _ = os.ReadFile(keyPath)
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

		// Only overwrite if the existing auth-provider is nil, or different from the newly specified one.
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

		// create new Exec if it does not already exist, otherwise just modify the command
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
			var newExecEnv []clientcmdapi.ExecEnvVar
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
			var newEnv []clientcmdapi.ExecEnvVar
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

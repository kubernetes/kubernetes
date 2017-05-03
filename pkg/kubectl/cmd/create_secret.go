/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"io"

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/kubectl"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/util/i18n"
)

// NewCmdCreateSecret groups subcommands to create various types of secrets
func NewCmdCreateSecret(f cmdutil.Factory, cmdOut, errOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "secret",
		Short: i18n.T("Create a secret using specified subcommand"),
		Long:  "Create a secret using specified subcommand.",
		Run:   cmdutil.DefaultSubCommandRun(errOut),
	}
	cmd.AddCommand(NewCmdCreateSecretDockerRegistry(f, cmdOut))
	cmd.AddCommand(NewCmdCreateSecretTLS(f, cmdOut))
	cmd.AddCommand(NewCmdCreateSecretGeneric(f, cmdOut))

	return cmd
}

var (
	secretLong = templates.LongDesc(i18n.T(`
		Create a secret based on a file, directory, or specified literal value.

		A single secret may package one or more key/value pairs.

		When creating a secret based on a file, the key will default to the basename of the file, and the value will
		default to the file content.  If the basename is an invalid key, you may specify an alternate key.

		When creating a secret based on a directory, each file whose basename is a valid key in the directory will be
		packaged into the secret.  Any directory entries except regular files are ignored (e.g. subdirectories,
		symlinks, devices, pipes, etc).`))

	secretExample = templates.Examples(i18n.T(`
	  # Create a new secret named my-secret with keys for each file in folder bar
	  kubectl create secret generic my-secret --from-file=path/to/bar

	  # Create a new secret named my-secret with specified keys instead of names on disk
	  kubectl create secret generic my-secret --from-file=ssh-privatekey=~/.ssh/id_rsa --from-file=ssh-publickey=~/.ssh/id_rsa.pub

	  # Create a new secret named my-secret with key1=supersecret and key2=topsecret
	  kubectl create secret generic my-secret --from-literal=key1=supersecret --from-literal=key2=topsecret

	  # Create a new secret named my-secret from an env file
	  kubectl create secret generic my-secret --from-env-file=path/to/bar.env`))
)

// NewCmdCreateSecretGeneric is a command to create generic secrets from files, directories, or literal values
func NewCmdCreateSecretGeneric(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "generic NAME [--type=string] [--from-file=[key=]source] [--from-literal=key1=value1] [--dry-run]",
		Short:   i18n.T("Create a secret from a local file, directory or literal value"),
		Long:    secretLong,
		Example: secretExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreateSecretGeneric(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.SecretV1GeneratorName)
	cmd.Flags().StringSlice("from-file", []string{}, "Key files can be specified using their file path, in which case a default name will be given to them, or optionally with a name and file path, in which case the given name will be used.  Specifying a directory will iterate each named file in the directory that is a valid secret key.")
	cmd.Flags().StringArray("from-literal", []string{}, "Specify a key and literal value to insert in secret (i.e. mykey=somevalue)")
	cmd.Flags().String("from-env-file", "", "Specify the path to a file to read lines of key=val pairs to create a secret (i.e. a Docker .env file).")
	cmd.Flags().String("type", "", i18n.T("The type of secret to create"))
	return cmd
}

// CreateSecretGeneric is the implementation of the create secret generic command
func CreateSecretGeneric(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.SecretV1GeneratorName:
		generator = &kubectl.SecretGeneratorV1{
			Name:           name,
			Type:           cmdutil.GetFlagString(cmd, "type"),
			FileSources:    cmdutil.GetFlagStringSlice(cmd, "from-file"),
			LiteralSources: cmdutil.GetFlagStringArray(cmd, "from-literal"),
			EnvFileSource:  cmdutil.GetFlagString(cmd, "from-env-file"),
		}
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("Generator: %s not supported.", generatorName))
	}
	return RunCreateSubcommand(f, cmd, cmdOut, &CreateSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetDryRunFlag(cmd),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}

var (
	secretForDockerRegistryLong = templates.LongDesc(i18n.T(`
		Create a new secret for use with Docker registries.

		Dockercfg secrets are used to authenticate against Docker registries.

		When using the Docker command line to push images, you can authenticate to a given registry by running

		    $ docker login DOCKER_REGISTRY_SERVER --username=DOCKER_USER --password=DOCKER_PASSWORD --email=DOCKER_EMAIL'.

    That produces a ~/.dockercfg file that is used by subsequent 'docker push' and 'docker pull' commands to
		authenticate to the registry. The email address is optional.

		When creating applications, you may have a Docker registry that requires authentication.  In order for the
		nodes to pull images on your behalf, they have to have the credentials.  You can provide this information
		by creating a dockercfg secret and attaching it to your service account.`))

	secretForDockerRegistryExample = templates.Examples(i18n.T(`
		  # If you don't already have a .dockercfg file, you can create a dockercfg secret directly by using:
		  kubectl create secret docker-registry my-secret --docker-server=DOCKER_REGISTRY_SERVER --docker-username=DOCKER_USER --docker-password=DOCKER_PASSWORD --docker-email=DOCKER_EMAIL`))
)

// NewCmdCreateSecretDockerRegistry is a macro command for creating secrets to work with Docker registries
func NewCmdCreateSecretDockerRegistry(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "docker-registry NAME --docker-username=user --docker-password=password --docker-email=email [--docker-server=string] [--from-literal=key1=value1] [--dry-run]",
		Short:   i18n.T("Create a secret for use with a Docker registry"),
		Long:    secretForDockerRegistryLong,
		Example: secretForDockerRegistryExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreateSecretDockerRegistry(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.SecretForDockerRegistryV1GeneratorName)
	cmd.Flags().String("docker-username", "", i18n.T("Username for Docker registry authentication"))
	cmd.MarkFlagRequired("docker-username")
	cmd.Flags().String("docker-password", "", i18n.T("Password for Docker registry authentication"))
	cmd.MarkFlagRequired("docker-password")
	cmd.Flags().String("docker-email", "", i18n.T("Email for Docker registry"))
	cmd.Flags().String("docker-server", "https://index.docker.io/v1/", i18n.T("Server location for Docker registry"))
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

// CreateSecretDockerRegistry is the implementation of the create secret docker-registry command
func CreateSecretDockerRegistry(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	requiredFlags := []string{"docker-username", "docker-password", "docker-email", "docker-server"}
	for _, requiredFlag := range requiredFlags {
		if value := cmdutil.GetFlagString(cmd, requiredFlag); len(value) == 0 {
			return cmdutil.UsageError(cmd, "flag %s is required", requiredFlag)
		}
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.SecretForDockerRegistryV1GeneratorName:
		generator = &kubectl.SecretForDockerRegistryGeneratorV1{
			Name:     name,
			Username: cmdutil.GetFlagString(cmd, "docker-username"),
			Email:    cmdutil.GetFlagString(cmd, "docker-email"),
			Password: cmdutil.GetFlagString(cmd, "docker-password"),
			Server:   cmdutil.GetFlagString(cmd, "docker-server"),
		}
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("Generator: %s not supported.", generatorName))
	}
	return RunCreateSubcommand(f, cmd, cmdOut, &CreateSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetDryRunFlag(cmd),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}

var (
	secretForTLSLong = templates.LongDesc(i18n.T(`
		Create a TLS secret from the given public/private key pair.

		The public/private key pair must exist before hand. The public key certificate must be .PEM encoded and match the given private key.`))

	secretForTLSExample = templates.Examples(i18n.T(`
	  # Create a new TLS secret named tls-secret with the given key pair:
	  kubectl create secret tls tls-secret --cert=path/to/tls.cert --key=path/to/tls.key`))
)

// NewCmdCreateSecretTLS is a macro command for creating secrets to work with Docker registries
func NewCmdCreateSecretTLS(f cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "tls NAME --cert=path/to/cert/file --key=path/to/key/file [--dry-run]",
		Short:   i18n.T("Create a TLS secret"),
		Long:    secretForTLSLong,
		Example: secretForTLSExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := CreateSecretTLS(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.SecretForTLSV1GeneratorName)
	cmd.Flags().String("cert", "", i18n.T("Path to PEM encoded public key certificate."))
	cmd.Flags().String("key", "", i18n.T("Path to private key associated with given certificate."))
	return cmd
}

// CreateSecretTLS is the implementation of the create secret tls command
func CreateSecretTLS(f cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	requiredFlags := []string{"cert", "key"}
	for _, requiredFlag := range requiredFlags {
		if value := cmdutil.GetFlagString(cmd, requiredFlag); len(value) == 0 {
			return cmdutil.UsageError(cmd, "flag %s is required", requiredFlag)
		}
	}
	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.SecretForTLSV1GeneratorName:
		generator = &kubectl.SecretForTLSGeneratorV1{
			Name: name,
			Key:  cmdutil.GetFlagString(cmd, "key"),
			Cert: cmdutil.GetFlagString(cmd, "cert"),
		}
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("Generator: %s not supported.", generatorName))
	}
	return RunCreateSubcommand(f, cmd, cmdOut, &CreateSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetFlagBool(cmd, "dry-run"),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}

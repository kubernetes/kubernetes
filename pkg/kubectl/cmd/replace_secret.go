/*
Copyright 2016 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/kubectl"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/resource"
)

// NewCmdReplaceSecret groups subcommands to replace various types of secrets
func NewCmdReplaceSecret(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:   "secret",
		Short: "Replacea secret using specified subcommand",
		Long:  "Replace a secret using specified subcommand.",
		Run: func(cmd *cobra.Command, args []string) {
			cmd.Help()
		},
	}
	cmd.AddCommand(NewCmdReplaceSecretDockerRegistry(f, cmdOut))
	cmd.AddCommand(NewCmdReplaceSecretTLS(f, cmdOut))
	cmd.AddCommand(NewCmdReplaceSecretGeneric(f, cmdOut))

	return cmd
}

var (
	replaceSecretLong = dedent.Dedent(`
		Replace a secret based on a file, directory, or specified literal value.

		A single secret may package one or more key/value pairs.

		When replacing a secret based on a file, the key will default to the basename of the file, and the value will
		default to the file content.  If the basename is an invalid key, you may specify an alternate key.

		When replacing a secret based on a directory, each file whose basename is a valid key in the directory will be
		packaged into the secret.  Any directory entries except regular files are ignored (e.g. subdirectories,
		symlinks, devices, pipes, etc).
		`)

	replaceSecretExample = dedent.Dedent(`
		  # Replace a secret named my-secret with keys for each file in folder bar
		  kubectl replace secret generic my-secret --from-file=path/to/bar

		  # Replace a secret named my-secret with specified keys instead of names on disk
		  kubectl replace secret generic my-secret --from-file=ssh-privatekey=~/.ssh/id_rsa --from-file=ssh-publickey=~/.ssh/id_rsa.pub

		  # Replace a secret named my-secret with key1=supersecret and key2=topsecret
		  kubectl replace secret generic my-secret --from-literal=key1=supersecret --from-literal=key2=topsecret`)
)

// NewCmdReplaceSecretGeneric is a command to replace generic secrets from files, directories, or literal values
func NewCmdReplaceSecretGeneric(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "generic NAME [--from-file=[key=]source] [--from-literal=key1=value1] [--dry-run]",
		Short:   "Replace a secret from a local file, directory or literal value",
		Long:    replaceSecretLong,
		Example: replaceSecretExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := ReplaceSecretGeneric(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.SecretV1GeneratorName)
	cmd.Flags().StringSlice("from-file", []string{}, "Key files can be specified using their file path, in which case a default name will be given to them, or optionally with a name and file path, in which case the given name will be used.  Specifying a directory will iterate each named file in the directory that is a valid secret key.")
	cmd.Flags().StringSlice("from-literal", []string{}, "Specify a key and literal value to insert in secret (i.e. mykey=somevalue)")
	return cmd
}

// ReplaceSecretGeneric is the implementation of the replace secret generic command
func ReplaceSecretGeneric(f *cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
	name, err := NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	cmdNamespace, _, err := f.DefaultNamespace()
	if err != nil {
		return err
	}
	mapper, typer := f.Object(cmdutil.GetIncludeThirdPartyAPIs(cmd))
	r := resource.NewBuilder(mapper, typer, resource.ClientMapperFunc(f.ClientForMapping), f.Decoder(true)).
		NamespaceParam(cmdNamespace).DefaultNamespace().
		ResourceTypeOrNameArgs(true, "secret", name).
		ContinueOnError().
		Latest().
		Flatten().
		Do()
	err = r.Err()
	if err != nil {
		return err
	}
	infos, err := r.Infos()
	if len(infos) != 1 {
		// There should be only one secret
		return fmt.Errorf("Expected 1 secret named \"%s\", but found %d", name, len(infos))
	}
	var oldSecretType api.SecretType
	switch obj := infos[0].Object.(type) {
	case *api.Secret:
		oldSecretType = obj.Type
	default:
		// This should not happen
		return errors.New("The type is expected to be secret")
	}

	var generator kubectl.StructuredGenerator
	switch generatorName := cmdutil.GetFlagString(cmd, "generator"); generatorName {
	case cmdutil.SecretV1GeneratorName:
		generator = &kubectl.SecretGeneratorV1{
			Name:           name,
			Type:           string(oldSecretType),
			FileSources:    cmdutil.GetFlagStringSlice(cmd, "from-file"),
			LiteralSources: cmdutil.GetFlagStringSlice(cmd, "from-literal"),
		}
	default:
		return cmdutil.UsageError(cmd, fmt.Sprintf("Generator: %s not supported.", generatorName))
	}
	return RunReplaceSubcommand(f, cmd, cmdOut, &ReplaceSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetDryRunFlag(cmd),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}

var (
	replaceSecretForDockerRegistryLong = dedent.Dedent(`
		Replace a secret for use with Docker registries.

		Dockercfg secrets are used to authenticate against Docker registries.

		When using the Docker command line to push images, you can authenticate to a given registry by running
		  'docker login DOCKER_REGISTRY_SERVER --username=DOCKER_USER --password=DOCKER_PASSWORD --email=DOCKER_EMAIL'.
		That produces a ~/.dockercfg file that is used by subsequent 'docker push' and 'docker pull' commands to
		authenticate to the registry.`)

	replaceSecretForDockerRegistryExample = dedent.Dedent(`
		  # If you already have a .dockercfg file, you can replace a dockercfg secret directly by using:
		  kubectl replace secret docker-registry my-secret --docker-server=DOCKER_REGISTRY_SERVER --docker-username=DOCKER_USER --docker-password=DOCKER_PASSWORD --docker-email=DOCKER_EMAIL`)
)

// NewCmdReplaceSecretDockerRegistry is a macro command for replacing secrets to work with Docker registries
func NewCmdReplaceSecretDockerRegistry(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "docker-registry NAME --docker-username=user --docker-password=password --docker-email=email [--docker-server=string] [--from-literal=key1=value1] [--dry-run]",
		Short:   "Replace a secret for use with a Docker registry",
		Long:    replaceSecretForDockerRegistryLong,
		Example: replaceSecretForDockerRegistryExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := ReplaceSecretDockerRegistry(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.SecretForDockerRegistryV1GeneratorName)
	cmd.Flags().String("docker-username", "", "Username for Docker registry authentication")
	cmd.MarkFlagRequired("docker-username")
	cmd.Flags().String("docker-password", "", "Password for Docker registry authentication")
	cmd.MarkFlagRequired("docker-password")
	cmd.Flags().String("docker-email", "", "Email for Docker registry")
	cmd.MarkFlagRequired("docker-email")
	cmd.Flags().String("docker-server", "https://index.docker.io/v1/", "Server location for Docker registry")
	cmdutil.AddInclude3rdPartyFlags(cmd)
	return cmd
}

// ReplaceSecretDockerRegistry is the implementation of the replace secret docker-registry command
func ReplaceSecretDockerRegistry(f *cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
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
	return RunReplaceSubcommand(f, cmd, cmdOut, &ReplaceSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetDryRunFlag(cmd),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}

var (
	replaceSecretForTLSLong = dedent.Dedent(`
		Replace a TLS secret from the given public/private key pair.

		The public/private key pair must exist before hand. The public key certificate must be .PEM encoded and match the given private key.`)

	replaceSecretForTLSExample = dedent.Dedent(`
		  # Replace a TLS secret named tls-secret with the given key pair:
		  kubectl replace secret tls tls-secret --cert=path/to/tls.cert --key=path/to/tls.key`)
)

// NewCmdReplaceSecretTLS is a macro command for replacing secret tls
func NewCmdReplaceSecretTLS(f *cmdutil.Factory, cmdOut io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "tls NAME --cert=path/to/cert/file --key=path/to/key/file [--dry-run]",
		Short:   "Replace a TLS secret",
		Long:    replaceSecretForTLSLong,
		Example: replaceSecretForTLSExample,
		Run: func(cmd *cobra.Command, args []string) {
			err := ReplaceSecretTLS(f, cmdOut, cmd, args)
			cmdutil.CheckErr(err)
		},
	}
	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.SecretForTLSV1GeneratorName)
	cmd.Flags().String("cert", "", "Path to PEM encoded public key certificate.")
	cmd.Flags().String("key", "", "Path to private key associated with given certificate.")
	return cmd
}

// ReplaceSecretTLS is the implementation of the replace secret tls command
func ReplaceSecretTLS(f *cmdutil.Factory, cmdOut io.Writer, cmd *cobra.Command, args []string) error {
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
	return RunReplaceSubcommand(f, cmd, cmdOut, &ReplaceSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              cmdutil.GetFlagBool(cmd, "dry-run"),
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}

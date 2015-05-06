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

package secrets

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/credentialprovider"
	cmdutil "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd/util"

	"github.com/spf13/cobra"
)

const (
	CreateDockerConfigSecretRecommendedName = "create-dockercfg"
)

type CreateDockerConfigOptions struct {
	SecretNamespace  string
	SecretName       string
	RegistryLocation string
	Username         string
	Password         string
	EmailAddress     string

	SecretsInterface client.SecretsInterface

	Out io.Writer
}

// NewCmdCreateDockerConfigSecret creates a command object for making a dockercfg secret
func NewCmdCreateDockerConfigSecret(name, fullName string, f *cmdutil.Factory, out io.Writer) *cobra.Command {
	o := &CreateDockerConfigOptions{Out: out}

	cmd := &cobra.Command{
		Use:   name + " <secret name>",
		Short: "Create a new dockercfg secret",
		Run: func(cmd *cobra.Command, args []string) {
			if err := o.Complete(f, args); err != nil {
				cmdutil.CheckErr(err)
			}

			if err := o.Validate(); err != nil {
				cmdutil.CheckErr(err)
			}

			if err := o.CreateDockerSecret(); err != nil {
				cmdutil.CheckErr(err)
			}

		},
	}

	cmd.Flags().StringVar(&o.Username, "docker-username", "", "username for docker registry authentication")
	cmd.Flags().StringVar(&o.Password, "docker-password", "", "password for docker registry authentication")
	cmd.Flags().StringVar(&o.EmailAddress, "docker-email", "", "email for docker registry")
	cmd.Flags().StringVar(&o.RegistryLocation, "docker-server", "https://index.docker.io/v1/", "server location for docker registry")

	return cmd
}

func (o CreateDockerConfigOptions) CreateDockerSecret() error {
	if err := o.Validate(); err != nil {
		return err
	}

	dockercfgAuth := credentialprovider.DockerConfigEntry{
		Username: o.Username,
		Password: o.Password,
		Email:    o.EmailAddress,
	}.ConvertToDockerConfigCompatible()

	dockerCfg := map[string]credentialprovider.DockerConfigEntryWithAuth{o.RegistryLocation: dockercfgAuth}

	dockercfgContent, err := json.Marshal(dockerCfg)
	if err != nil {
		return err
	}

	secret := &api.Secret{}
	secret.Namespace = o.SecretNamespace
	secret.Name = o.SecretName
	secret.Type = api.SecretTypeDockercfg
	secret.Data = map[string][]byte{}
	secret.Data[api.DockerConfigKey] = dockercfgContent

	if _, err := o.SecretsInterface.Create(secret); err != nil {
		return err
	}

	fmt.Fprintf(o.GetOut(), "%s/%s\n", secret.Namespace, secret.Name)

	return nil
}

func (o *CreateDockerConfigOptions) Complete(f *cmdutil.Factory, args []string) error {
	if len(args) != 1 {
		return errors.New("must have exactly one argument: secret name")
	}
	o.SecretName = args[0]

	client, err := f.Client()
	if err != nil {
		return err
	}
	o.SecretNamespace, err = f.DefaultNamespace()
	if err != nil {
		return err
	}

	o.SecretsInterface = client.Secrets(o.SecretNamespace)

	return nil
}

func (o CreateDockerConfigOptions) Validate() error {
	if len(o.SecretNamespace) == 0 {
		return errors.New("SecretNamespace must be present")
	}
	if len(o.SecretName) == 0 {
		return errors.New("secret name must be present")
	}
	if len(o.RegistryLocation) == 0 {
		return errors.New("docker-server must be present")
	}
	if len(o.Username) == 0 {
		return errors.New("docker-username must be present")
	}
	if len(o.Password) == 0 {
		return errors.New("docker-password must be present")
	}
	if len(o.EmailAddress) == 0 {
		return errors.New("docker-email must be present")
	}
	if o.SecretsInterface == nil {
		return errors.New("SecretsInterface must be present")
	}

	if strings.Contains(o.Username, ":") {
		return fmt.Errorf("username '%v' is illegal because it contains a ':'", o.Username)
	}

	return nil
}

func (o CreateDockerConfigOptions) GetOut() io.Writer {
	if o.Out == nil {
		return ioutil.Discard
	}

	return o.Out
}

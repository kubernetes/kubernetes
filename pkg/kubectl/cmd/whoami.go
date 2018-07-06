/*
Copyright 2018 The Kubernetes Authors.

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
	"encoding/json"
	"errors"
	"fmt"
	"strings"

	"github.com/ghodss/yaml"
	"github.com/spf13/cobra"
	"k8s.io/apiserver/pkg/authentication/user"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/kubectl/genericclioptions"
	"k8s.io/kubernetes/pkg/kubectl/util/i18n"
)

var (
	whoamiExample = templates.Examples(i18n.T(`
		# Print the user identity information from the client's' request 
		kubectl whoami`))
)

type WhoAmIOptions struct {
	Output string

	genericclioptions.IOStreams
	Client *restclient.RESTClient
}

func NewWhoAmIOptions(ioStreams genericclioptions.IOStreams) *WhoAmIOptions {
	return &WhoAmIOptions{
		IOStreams: ioStreams,
	}
}

func NewCmdWhoAmI(f cmdutil.Factory, ioStreams genericclioptions.IOStreams) *cobra.Command {
	o := NewWhoAmIOptions(ioStreams)
	cmd := &cobra.Command{
		Use:     "whoami",
		Short:   i18n.T("Print the user's identity information"),
		Long:    "Print the requesting user's identity for the current context",
		Example: whoamiExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.Run())
		},
	}
	cmd.Flags().StringVarP(&o.Output, "output", "o", o.Output, "One of 'yaml' or 'json'.")
	return cmd
}

func (o *WhoAmIOptions) Complete(f cmdutil.Factory, cmd *cobra.Command) error {
	restClient, err := f.RESTClient()
	if err != nil {
		return err
	}
	o.Client = restClient
	return nil
}

func (o *WhoAmIOptions) Validate() error {
	if o.Output != "" && o.Output != "yaml" && o.Output != "json" {
		return errors.New(`--output must be 'yaml' or 'json'`)
	}

	return nil
}

func (o *WhoAmIOptions) Run() error {
	rawUserInfo, err := o.Client.Get().AbsPath("/whoami").DoRaw()
	if err != nil {
		return err
	}

	userInfo := user.DefaultInfo{}
	if err := json.Unmarshal(rawUserInfo, &userInfo); err != nil {
		return err
	}

	switch o.Output {
	case "":
		o.IOStreams.Out.Write([]byte(fmt.Sprintf("User Name: %s\n", userInfo.Name)))
		o.IOStreams.Out.Write([]byte(fmt.Sprintf("User Groups: %s\n", strings.Join(userInfo.Groups, ", "))))
	case "yaml":
		marshalled, err := yaml.Marshal(&userInfo)
		if err != nil {
			return err
		}
		o.IOStreams.Out.Write(marshalled)
	case "json":
		marshalled, err := json.MarshalIndent(&userInfo, "", "  ")
		if err != nil {
			return err
		}
		o.IOStreams.Out.Write(marshalled)
	}
	return nil
}

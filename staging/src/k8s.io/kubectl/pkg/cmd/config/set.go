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
	"reflect"

	"github.com/spf13/cobra"

	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	clientcmdapiv1 "k8s.io/client-go/tools/clientcmd/api/v1"
	cliflag "k8s.io/component-base/cli/flag"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

type setOptions struct {
	configAccess  clientcmd.ConfigAccess
	propertyName  string
	propertyValue string
	setRawBytes   cliflag.Tristate
	jsonPath      bool
	deduplicate   bool

	streams genericclioptions.IOStreams
}

var (
	setLong = templates.LongDesc(i18n.T(`
	Set an individual value in a kubeconfig file.

	PROPERTY_NAME is either a jsonpath query for where the properties will be, or a dot delimited name.

	PROPERTY_VALUE is the new value you want to set. Binary fields such as 'certificate-authority-data' expect a base64 encoded string unless the --set-raw-bytes flag is used.

	Specifying an attribute name that already exists will merge new fields on top of existing values.`))

	setExample = templates.Examples(`
	# Set the server field on the my-cluster cluster to https://1.2.3.4
	kubectl config set clusters.my-cluster.server https://1.2.3.4

	# Set the certificate-authority-data field on the my-cluster cluster
	kubectl config set clusters.my-cluster.certificate-authority-data $(echo "cert_data_here" | base64 -i -)

	# Set the cluster field in the my-context context to my-cluster
	kubectl config set contexts.my-context.cluster my-cluster

	# Set the client-key-data field in the cluster-admin user using --set-raw-bytes option
	kubectl config set users.cluster-admin.client-key-data cert_data_here --set-raw-bytes=true

	# Set the server for a cluster with name cluster-0 using jsonpath
	kubectl config set '{.clusters[?(@.name=="cluster-0")].cluster.server}' "https://1.2.3.4"

	# Set the same username value for all users using the wildcard filter
	kubectl config set '{.users[*].user.username}' "test-user"

	# Set a new list using users[*].user.exec.args with jsonpath
	kubectl config set '{.users[*].user.exec.args}' "arg1,arg2"

	# Add a list item using users[*].user.exec.args with jsonpath
	kubectl config set '{.users[*].user.exec.args}' "arg3+"

	# Remove a list item using users[*].user.exec.args with jsonpath
	kubectl config set '{.users[*].user.exec.args}' "arg2-"

	# Set new list that will be deduplicated and sorted, will result in list of arg1,arg2,arg3,arg4
	kubectl config set '{.users[*].user.exec.args}' "arg1,arg2,agr2,arg4,arg3" --deduplicate

	# Deduplicate and sort existing list without making any updates to it
	kubectl config set '{.users[*].user.exec.args}' "+" --deduplicate`)
)

// NewCmdConfigSet returns a Command instance for 'config set' sub command
func NewCmdConfigSet(streams genericclioptions.IOStreams, configAccess clientcmd.ConfigAccess) *cobra.Command {
	options := &setOptions{configAccess: configAccess, jsonPath: false, streams: streams}

	cmd := &cobra.Command{
		Use:                   "set PROPERTY_NAME PROPERTY_VALUE",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Set an individual value in a kubeconfig file"),
		Long:                  setLong,
		Example:               setExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(options.complete(cmd))
			cmdutil.CheckErr(options.run())
			fmt.Fprintf(options.streams.Out, "Property %q set.\n", options.propertyName)
		},
	}

	f := cmd.Flags().VarPF(&options.setRawBytes, "set-raw-bytes", "", "When writing a []byte PROPERTY_VALUE, write the given string directly without base64 decoding.")
	f.NoOptDefVal = "true"
	cmd.Flags().BoolVar(&options.deduplicate, "deduplicate", false, "Whether to use deduplicate list of values or not. This flag will also sort the list.")

	return cmd
}

func (o setOptions) run() error {
	err := o.validate()
	if err != nil {
		return err
	}

	config, err := o.configAccess.GetStartingConfig()
	if err != nil {
		return err
	}

	if o.jsonPath {
		// Convert api config to apiv1 config, so we can use jsonpath properly
		v1Config := &clientcmdapiv1.Config{}
		if err := clientcmdapiv1.Convert_api_Config_To_v1_Config(config, v1Config, nil); err != nil {
			return err
		}

		if err := modifyConfigJson(v1Config, o.propertyName, o.propertyValue, false, o.setRawBytes.Value(), o.deduplicate); err != nil {
			return err
		}

		// Convert the apiv1 config back to an api config to write back out
		finalConfig := clientcmdapi.NewConfig()
		if err := clientcmdapiv1.Convert_v1_Config_To_api_Config(v1Config, finalConfig, nil); err != nil {
			return err
		}
		config = finalConfig
	} else {
		if _, err := fmt.Fprintln(o.streams.ErrOut, "Warning: usage of dot delimited path for setting config values is deprecated, please use jsonpath syntax instead."); err != nil {
			return fmt.Errorf("failed to write warning message to user")
		}
		steps, err := newNavigationSteps(o.propertyName)
		if err != nil {
			return err
		}

		setRawBytes := false
		if o.setRawBytes.Provided() {
			setRawBytes = o.setRawBytes.Value()
		}

		err = modifyConfig(reflect.ValueOf(config), steps, o.propertyValue, false, setRawBytes)
		if err != nil {
			return err
		}
	}

	if err := clientcmd.ModifyConfig(o.configAccess, *config, false); err != nil {
		return err
	}

	return nil
}

func (o *setOptions) complete(cmd *cobra.Command) error {
	endingArgs := cmd.Flags().Args()
	if len(endingArgs) != 2 {
		return helpErrorf(cmd, "Unexpected args: %v", endingArgs)
	}

	o.propertyValue = endingArgs[1]
	o.propertyName = endingArgs[0]

	// try to determine if we have a jsonpath from first character of first argument
	// this should only ever be a { if this is a jsonpath string, if it is not we will try to use the old dot delimited
	// syntax instead.
	if string(endingArgs[0][0]) == "{" {
		o.jsonPath = true
	}
	return nil
}

func (o setOptions) validate() error {
	if len(o.propertyValue) == 0 {
		return errors.New("you cannot use set to unset a property")
	}

	if len(o.propertyName) == 0 {
		return errors.New("you must specify a property")
	}

	return nil
}

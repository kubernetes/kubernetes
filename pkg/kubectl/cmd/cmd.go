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

package cmd

import (
	"fmt"
	"io"
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"

	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

// Factory provides abstractions that allow the Kubectl command to be extended across multiple types
// of resources and different API sets.
type Factory struct {
	ClientBuilder clientcmd.Builder
	Mapper        meta.RESTMapper
	Typer         runtime.ObjectTyper
	Client        func(cmd *cobra.Command, mapping *meta.RESTMapping) (kubectl.RESTClient, error)
	Describer     func(cmd *cobra.Command, mapping *meta.RESTMapping) (kubectl.Describer, error)
	Printer       func(cmd *cobra.Command, mapping *meta.RESTMapping, noHeaders bool) (kubectl.ResourcePrinter, error)
	Validator     func(*cobra.Command) (validation.Schema, error)
}

// NewFactory creates a factory with the default Kubernetes resources defined
func NewFactory(clientBuilder clientcmd.Builder) *Factory {
	return &Factory{
		ClientBuilder: clientBuilder,
		Mapper:        latest.RESTMapper,
		Typer:         api.Scheme,
		Validator: func(cmd *cobra.Command) (validation.Schema, error) {
			if GetFlagBool(cmd, "validate") {
				client, err := clientBuilder.Client()
				if err != nil {
					return nil, err
				}
				return &clientSwaggerSchema{client, api.Scheme}, nil
			} else {
				return validation.NullSchema{}, nil
			}
		},
		Client: func(cmd *cobra.Command, mapping *meta.RESTMapping) (kubectl.RESTClient, error) {
			return clientBuilder.Client()
		},
		Describer: func(cmd *cobra.Command, mapping *meta.RESTMapping) (kubectl.Describer, error) {
			client, err := clientBuilder.Client()
			if err != nil {
				return nil, err
			}
			describer, ok := kubectl.DescriberFor(mapping.Kind, client)
			if !ok {
				return nil, fmt.Errorf("no description has been implemented for %q", mapping.Kind)
			}
			return describer, nil
		},
		Printer: func(cmd *cobra.Command, mapping *meta.RESTMapping, noHeaders bool) (kubectl.ResourcePrinter, error) {
			return kubectl.NewHumanReadablePrinter(noHeaders), nil
		},
	}
}

func (f *Factory) Run(out io.Writer) {
	// Parent command to which all subcommands are added.
	cmds := &cobra.Command{
		Use:   "kubectl",
		Short: "kubectl controls the Kubernetes cluster manager",
		Long: `kubectl controls the Kubernetes cluster manager.

Find more information at https://github.com/GoogleCloudPlatform/kubernetes.`,
		Run: runHelp,
	}

	f.ClientBuilder.BindFlags(cmds.PersistentFlags())

	// Globally persistent flags across all subcommands.
	// TODO Change flag names to consts to allow safer lookup from subcommands.
	// TODO Add a verbose flag that turns on glog logging. Probably need a way
	// to do that automatically for every subcommand.
	cmds.PersistentFlags().String("ns_path", os.Getenv("HOME")+"/.kubernetes_ns", "Path to the namespace info file that holds the namespace context to use for CLI requests.")
	cmds.PersistentFlags().StringP("namespace", "n", "", "If present, the namespace scope for this CLI request.")
	cmds.PersistentFlags().Bool("validate", false, "If true, use a schema to validate the input before sending it")

	cmds.AddCommand(f.NewCmdVersion(out))
	cmds.AddCommand(f.NewCmdProxy(out))

	cmds.AddCommand(f.NewCmdGet(out))
	cmds.AddCommand(f.NewCmdDescribe(out))
	cmds.AddCommand(f.NewCmdCreate(out))
	cmds.AddCommand(f.NewCmdCreateAll(out))
	cmds.AddCommand(f.NewCmdUpdate(out))
	cmds.AddCommand(f.NewCmdDelete(out))

	cmds.AddCommand(NewCmdNamespace(out))
	cmds.AddCommand(f.NewCmdLog(out))

	if err := cmds.Execute(); err != nil {
		os.Exit(1)
	}
}

func checkErr(err error) {
	if err != nil {
		glog.FatalDepth(1, err)
	}
}

func usageError(cmd *cobra.Command, format string, args ...interface{}) {
	glog.Errorf(format, args...)
	glog.Errorf("See '%s -h' for help.", cmd.CommandPath())
	os.Exit(1)
}

func runHelp(cmd *cobra.Command, args []string) {
	cmd.Help()
}

// GetKubeNamespace returns the value of the namespace a
// user provided on the command line or use the default
// namespace.
func GetKubeNamespace(cmd *cobra.Command) string {
	result := api.NamespaceDefault
	if ns := GetFlagString(cmd, "namespace"); len(ns) > 0 {
		result = ns
		glog.V(2).Infof("Using namespace from -ns flag")
	} else {
		nsPath := GetFlagString(cmd, "ns_path")
		nsInfo, err := kubectl.LoadNamespaceInfo(nsPath)
		if err != nil {
			glog.Fatalf("Error loading current namespace: %v", err)
		}
		result = nsInfo.Namespace
	}
	glog.V(2).Infof("Using namespace %s", result)
	return result
}

// GetExplicitKubeNamespace returns the value of the namespace a
// user explicitly provided on the command line, or false if no
// such namespace was specified.
func GetExplicitKubeNamespace(cmd *cobra.Command) (string, bool) {
	if ns := GetFlagString(cmd, "namespace"); len(ns) > 0 {
		return ns, true
	}
	// TODO: determine when --ns_path is set but equal to the default
	// value and return its value and true.
	return "", false
}

type clientSwaggerSchema struct {
	c *client.Client
	t runtime.ObjectTyper
}

func (c *clientSwaggerSchema) ValidateBytes(data []byte) error {
	version, _, err := c.t.DataVersionAndKind(data)
	if err != nil {
		return err
	}
	schemaData, err := c.c.RESTClient.Get().
		AbsPath("/swaggerapi/api").
		Path(version).
		Do().
		Raw()
	if err != nil {
		return err
	}
	schema, err := validation.NewSwaggerSchemaFromBytes(schemaData)
	if err != nil {
		return err
	}
	return schema.ValidateBytes(data)
}

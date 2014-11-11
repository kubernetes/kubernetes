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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

// Factory provides abstractions that allow the Kubectl command to be extended across multiple types
// of resources and different API sets.
type Factory struct {
	Mapper    meta.RESTMapper
	Typer     runtime.ObjectTyper
	Client    func(*cobra.Command, *meta.RESTMapping) (kubectl.RESTClient, error)
	Describer func(*cobra.Command, *meta.RESTMapping) (kubectl.Describer, error)
	Printer   func(cmd *cobra.Command, mapping *meta.RESTMapping, noHeaders bool) (kubectl.ResourcePrinter, error)
}

// NewFactory creates a factory with the default Kubernetes resources defined
func NewFactory() *Factory {
	return &Factory{
		Mapper: latest.RESTMapper,
		Typer:  api.Scheme,
		Client: func(cmd *cobra.Command, mapping *meta.RESTMapping) (kubectl.RESTClient, error) {
			return getKubeClient(cmd), nil
		},
		Describer: func(cmd *cobra.Command, mapping *meta.RESTMapping) (kubectl.Describer, error) {
			describer, ok := kubectl.DescriberFor(mapping.Kind, getKubeClient(cmd))
			if !ok {
				return nil, fmt.Errorf("No description has been implemented for %q", mapping.Kind)
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

	// Globally persistent flags across all subcommands.
	// TODO Change flag names to consts to allow safer lookup from subcommands.
	// TODO Add a verbose flag that turns on glog logging. Probably need a way
	// to do that automatically for every subcommand.
	cmds.PersistentFlags().StringP("server", "s", "", "Kubernetes apiserver to connect to")
	cmds.PersistentFlags().StringP("auth-path", "a", os.Getenv("HOME")+"/.kubernetes_auth", "Path to the auth info file. If missing, prompt the user. Only used if using https.")
	cmds.PersistentFlags().Bool("match-server-version", false, "Require server version to match client version")
	cmds.PersistentFlags().String("api-version", latest.Version, "The version of the API to use against the server")
	cmds.PersistentFlags().String("certificate-authority", "", "Path to a certificate file for the certificate authority")
	cmds.PersistentFlags().String("client-certificate", "", "Path to a client certificate for TLS.")
	cmds.PersistentFlags().String("client-key", "", "Path to a client key file for TLS.")
	cmds.PersistentFlags().Bool("insecure-skip-tls-verify", false, "If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.")
	cmds.PersistentFlags().String("ns-path", os.Getenv("HOME")+"/.kubernetes_ns", "Path to the namespace info file that holds the namespace context to use for CLI requests.")
	cmds.PersistentFlags().StringP("namespace", "n", "", "If present, the namespace scope for this CLI request.")

	cmds.AddCommand(NewCmdVersion(out))
	cmds.AddCommand(NewCmdProxy(out))

	cmds.AddCommand(f.NewCmdGet(out))
	cmds.AddCommand(f.NewCmdDescribe(out))
	cmds.AddCommand(f.NewCmdCreate(out))
	cmds.AddCommand(f.NewCmdCreateAll(out))
	cmds.AddCommand(f.NewCmdUpdate(out))
	cmds.AddCommand(f.NewCmdDelete(out))

	cmds.AddCommand(NewCmdNamespace(out))
	cmds.AddCommand(NewCmdLog(out))

	if err := cmds.Execute(); err != nil {
		os.Exit(1)
	}
}

// TODO: remove this function and references to it-- errors it prints are
// very unhelpful because file/line number are wrong.
func checkErr(err error) {
	if err != nil {
		glog.Fatalf("%v", err)
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

func getKubeNamespace(cmd *cobra.Command) string {
	result := api.NamespaceDefault
	if ns := GetFlagString(cmd, "namespace"); len(ns) > 0 {
		result = ns
		glog.V(2).Infof("Using namespace from -ns flag")
	} else {
		nsPath := GetFlagString(cmd, "ns-path")
		nsInfo, err := kubectl.LoadNamespaceInfo(nsPath)
		if err != nil {
			glog.Fatalf("Error loading current namespace: %v", err)
		}
		result = nsInfo.Namespace
	}
	glog.V(2).Infof("Using namespace %s", result)
	return result
}

// getExplicitKubeNamespace returns the value of the namespace a
// user explicitly provided on the command line, or false if no
// such namespace was specified.
func getExplicitKubeNamespace(cmd *cobra.Command) (string, bool) {
	if ns := GetFlagString(cmd, "namespace"); len(ns) > 0 {
		return ns, true
	}
	// TODO: determine when --ns-path is set but equal to the default
	// value and return its value and true.
	return "", false
}

// GetKubeConfig returns a config used for the Kubernetes client with CLI
// options parsed.
func GetKubeConfig(cmd *cobra.Command) *client.Config {
	config := &client.Config{}

	var host string
	if hostFlag := GetFlagString(cmd, "server"); len(hostFlag) > 0 {
		host = hostFlag
		glog.V(2).Infof("Using server from -s flag: %s", host)
	} else if len(os.Getenv("KUBERNETES_MASTER")) > 0 {
		host = os.Getenv("KUBERNETES_MASTER")
		glog.V(2).Infof("Using server from env var KUBERNETES_MASTER: %s", host)
	} else {
		// TODO: eventually apiserver should start on 443 and be secure by default
		host = "http://localhost:8080"
		glog.V(2).Infof("No server found in flag or env var, using default: %s", host)
	}
	config.Host = host

	if client.IsConfigTransportSecure(config) {
		// Get the values from the file on disk (or from the user at the
		// command line). Override them with the command line parameters, if
		// provided.
		authPath := GetFlagString(cmd, "auth-path")
		authInfo, err := kubectl.LoadAuthInfo(authPath, os.Stdin)
		if err != nil {
			glog.Fatalf("Error loading auth: %v", err)
		}

		config.Username = authInfo.User
		config.Password = authInfo.Password
		// First priority is flag, then file.
		config.CAFile = FirstNonEmptyString(GetFlagString(cmd, "certificate-authority"), authInfo.CAFile)
		config.CertFile = FirstNonEmptyString(GetFlagString(cmd, "client-certificate"), authInfo.CertFile)
		config.KeyFile = FirstNonEmptyString(GetFlagString(cmd, "client-key"), authInfo.KeyFile)
		config.BearerToken = authInfo.BearerToken
		// For config.Insecure, the command line ALWAYS overrides the authInfo
		// file, regardless of its setting.
		if insecureFlag := GetFlagBoolPtr(cmd, "insecure-skip-tls-verify"); insecureFlag != nil {
			config.Insecure = *insecureFlag
		} else if authInfo.Insecure != nil {
			config.Insecure = *authInfo.Insecure
		}
	}

	// The API version (e.g. v1beta1), not the binary version.
	config.Version = GetFlagString(cmd, "api-version")

	return config
}

func getKubeClient(cmd *cobra.Command) *client.Client {
	config := GetKubeConfig(cmd)

	// The binary version.
	matchVersion := GetFlagBool(cmd, "match-server-version")

	c, err := kubectl.GetKubeClient(config, matchVersion)
	if err != nil {
		glog.Fatalf("Error creating kubernetes client: %v", err)
	}
	return c
}

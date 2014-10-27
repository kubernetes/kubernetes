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
	"io/ioutil"
	"net/http"
	"os"
	"strconv"
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/golang/glog"
	"github.com/spf13/cobra"
)

type Factory struct {
	Mapper meta.RESTMapper
	Typer  runtime.ObjectTyper
	Client func(*cobra.Command, *meta.RESTMapping) (kubectl.RESTClient, error)
}

func RunKubectl(out io.Writer) {
	// Parent command to which all subcommands are added.
	cmds := &cobra.Command{
		Use:   "kubectl",
		Short: "kubectl controls the Kubernetes cluster manager",
		Long: `kubectl controls the Kubernetes cluster manager.

Find more information at https://github.com/GoogleCloudPlatform/kubernetes.`,
		Run: runHelp,
	}

	factory := &Factory{
		Mapper: latest.NewDefaultRESTMapper(),
		Typer:  api.Scheme,
		Client: func(cmd *cobra.Command, mapping *meta.RESTMapping) (kubectl.RESTClient, error) {
			// Will handle all resources defined by the command
			return getKubeClient(cmd), nil
		},
	}

	// Globally persistent flags across all subcommands.
	// TODO Change flag names to consts to allow safer lookup from subcommands.
	// TODO Add a verbose flag that turns on glog logging. Probably need a way
	// to do that automatically for every subcommand.
	cmds.PersistentFlags().StringP("server", "s", "", "Kubernetes apiserver to connect to")
	cmds.PersistentFlags().StringP("auth-path", "a", os.Getenv("HOME")+"/.kubernetes_auth", "Path to the auth info file. If missing, prompt the user. Only used if using https.")
	cmds.PersistentFlags().Bool("match-server-version", false, "Require server version to match client version")
	cmds.PersistentFlags().String("api-version", latest.Version, "The version of the API to use against the server (used for viewing resources only)")
	cmds.PersistentFlags().String("certificate-authority", "", "Path to a certificate file for the certificate authority")
	cmds.PersistentFlags().String("client-certificate", "", "Path to a client certificate for TLS.")
	cmds.PersistentFlags().String("client-key", "", "Path to a client key file for TLS.")
	cmds.PersistentFlags().Bool("insecure-skip-tls-verify", false, "If true, the server's certificate will not be checked for validity. This will make your HTTPS connections insecure.")
	cmds.PersistentFlags().String("ns-path", os.Getenv("HOME")+"/.kubernetes_ns", "Path to the namespace info file that holds the namespace context to use for CLI requests.")
	cmds.PersistentFlags().StringP("namespace", "n", "", "If present, the namespace scope for this CLI request.")

	cmds.AddCommand(NewCmdVersion(out))
	cmds.AddCommand(NewCmdProxy(out))
	cmds.AddCommand(NewCmdGet(out))
	cmds.AddCommand(NewCmdDescribe(out))

	cmds.AddCommand(factory.NewCmdCreate(out))
	cmds.AddCommand(factory.NewCmdUpdate(out))
	cmds.AddCommand(factory.NewCmdDelete(out))

	cmds.AddCommand(NewCmdNamespace(out))
	cmds.AddCommand(NewCmdLog(out))
	cmds.AddCommand(NewCmdCreateAll(out))

	if err := cmds.Execute(); err != nil {
		os.Exit(1)
	}
}

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

func getFlagString(cmd *cobra.Command, flag string) string {
	f := cmd.Flags().Lookup(flag)
	if f == nil {
		glog.Fatalf("Flag accessed but not defined for command %s: %s", cmd.Name(), flag)
	}
	return f.Value.String()
}

func getFlagBool(cmd *cobra.Command, flag string) bool {
	f := cmd.Flags().Lookup(flag)
	if f == nil {
		glog.Fatalf("Flag accessed but not defined for command %s: %s", cmd.Name(), flag)
	}
	// Caseless compare.
	if strings.ToLower(f.Value.String()) == "true" {
		return true
	}
	return false
}

// Returns nil if the flag wasn't set.
func getFlagBoolPtr(cmd *cobra.Command, flag string) *bool {
	f := cmd.Flags().Lookup(flag)
	if f == nil {
		glog.Fatalf("Flag accessed but not defined for command %s: %s", cmd.Name(), flag)
	}
	// Check if flag was not set at all.
	if !f.Changed && f.DefValue == f.Value.String() {
		return nil
	}
	var ret bool
	// Caseless compare.
	if strings.ToLower(f.Value.String()) == "true" {
		ret = true
	} else {
		ret = false
	}
	return &ret
}

// Assumes the flag has a default value.
func getFlagInt(cmd *cobra.Command, flag string) int {
	f := cmd.Flags().Lookup(flag)
	if f == nil {
		glog.Fatalf("Flag accessed but not defined for command %s: %s", cmd.Name(), flag)
	}
	v, err := strconv.Atoi(f.Value.String())
	// This is likely not a sufficiently friendly error message, but cobra
	// should prevent non-integer values from reaching here.
	checkErr(err)
	return v
}

func getKubeNamespace(cmd *cobra.Command) string {
	result := api.NamespaceDefault
	if ns := getFlagString(cmd, "namespace"); len(ns) > 0 {
		result = ns
		glog.V(2).Infof("Using namespace from -ns flag")
	} else {
		nsPath := getFlagString(cmd, "ns-path")
		nsInfo, err := kubectl.LoadNamespaceInfo(nsPath)
		if err != nil {
			glog.Fatalf("Error loading current namespace: %v", err)
		}
		result = nsInfo.Namespace
	}
	glog.V(2).Infof("Using namespace %s", result)
	return result
}

func getKubeConfig(cmd *cobra.Command) *client.Config {
	config := &client.Config{}

	var host string
	if hostFlag := getFlagString(cmd, "server"); len(hostFlag) > 0 {
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
		authPath := getFlagString(cmd, "auth-path")
		authInfo, err := kubectl.LoadAuthInfo(authPath, os.Stdin)
		if err != nil {
			glog.Fatalf("Error loading auth: %v", err)
		}

		config.Username = authInfo.User
		config.Password = authInfo.Password
		// First priority is flag, then file.
		config.CAFile = firstNonEmptyString(getFlagString(cmd, "certificate-authority"), authInfo.CAFile)
		config.CertFile = firstNonEmptyString(getFlagString(cmd, "client-certificate"), authInfo.CertFile)
		config.KeyFile = firstNonEmptyString(getFlagString(cmd, "client-key"), authInfo.KeyFile)
		config.BearerToken = authInfo.BearerToken
		// For config.Insecure, the command line ALWAYS overrides the authInfo
		// file, regardless of its setting.
		if insecureFlag := getFlagBoolPtr(cmd, "insecure-skip-tls-verify"); insecureFlag != nil {
			config.Insecure = *insecureFlag
		} else if authInfo.Insecure != nil {
			config.Insecure = *authInfo.Insecure
		}
	}

	// The API version (e.g. v1beta1), not the binary version.
	config.Version = getFlagString(cmd, "api-version")

	return config
}

func getKubeClient(cmd *cobra.Command) *client.Client {
	config := getKubeConfig(cmd)

	// The binary version.
	matchVersion := getFlagBool(cmd, "match-server-version")

	c, err := kubectl.GetKubeClient(config, matchVersion)
	if err != nil {
		glog.Fatalf("Error creating kubernetes client: %v", err)
	}
	return c
}

// Returns the first non-empty string out of the ones provided. If all
// strings are empty, returns an empty string.
func firstNonEmptyString(args ...string) string {
	for _, s := range args {
		if len(s) > 0 {
			return s
		}
	}
	return ""
}

// readConfigData reads the bytes from the specified filesytem or network
// location or from stdin if location == "-".
func readConfigData(location string) ([]byte, error) {
	if len(location) == 0 {
		return nil, fmt.Errorf("Location given but empty")
	}

	if location == "-" {
		// Read from stdin.
		data, err := ioutil.ReadAll(os.Stdin)
		if err != nil {
			return nil, err
		}

		if len(data) == 0 {
			return nil, fmt.Errorf(`Read from stdin specified ("-") but no data found`)
		}

		return data, nil
	}

	// Use the location as a file path or URL.
	return readConfigDataFromLocation(location)
}

func readConfigDataFromLocation(location string) ([]byte, error) {
	// we look for http:// or https:// to determine if valid URL, otherwise do normal file IO
	if strings.Index(location, "http://") == 0 || strings.Index(location, "https://") == 0 {
		resp, err := http.Get(location)
		if err != nil {
			return nil, fmt.Errorf("Unable to access URL %s: %v\n", location, err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			return nil, fmt.Errorf("Unable to read URL, server reported %d %s", resp.StatusCode, resp.Status)
		}
		data, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("Unable to read URL %s: %v\n", location, err)
		}
		return data, nil
	} else {
		data, err := ioutil.ReadFile(location)
		if err != nil {
			return nil, fmt.Errorf("Unable to read %s: %v\n", location, err)
		}
		return data, nil
	}
}

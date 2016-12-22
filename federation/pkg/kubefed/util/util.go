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

package util

import (
	"k8s.io/kubernetes/pkg/api"
	client "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	kubectlcmd "k8s.io/kubernetes/pkg/kubectl/cmd"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"

	"github.com/spf13/cobra"
)

const (
	// KubeconfigSecretDataKey is the key name used in the secret to
	// stores a cluster's credentials.
	KubeconfigSecretDataKey = "kubeconfig"

	// DefaultFederationSystemNamespace is the namespace in which
	// federation system components are hosted.
	DefaultFederationSystemNamespace = "federation-system"
)

// AdminConfig provides a filesystem based kubeconfig (via
// `PathOptions()`) and a mechanism to talk to the federation
// host cluster.
type AdminConfig interface {
	// PathOptions provides filesystem based kubeconfig access.
	PathOptions() *clientcmd.PathOptions
	// HostFactory provides a mechanism to communicate with the
	// cluster where federation control plane is hosted.
	HostFactory(host, kubeconfigPath string) cmdutil.Factory
}

// adminConfig implements the AdminConfig interface.
type adminConfig struct {
	pathOptions *clientcmd.PathOptions
}

// NewAdminConfig creates an admin config for `kubefed` commands.
func NewAdminConfig(pathOptions *clientcmd.PathOptions) AdminConfig {
	return &adminConfig{
		pathOptions: pathOptions,
	}
}

func (a *adminConfig) PathOptions() *clientcmd.PathOptions {
	return a.pathOptions
}

func (a *adminConfig) HostFactory(host, kubeconfigPath string) cmdutil.Factory {
	loadingRules := *a.pathOptions.LoadingRules
	loadingRules.Precedence = a.pathOptions.GetLoadingPrecedence()
	loadingRules.ExplicitPath = kubeconfigPath
	overrides := &clientcmd.ConfigOverrides{
		CurrentContext: host,
	}

	hostClientConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(&loadingRules, overrides)

	return cmdutil.NewFactory(hostClientConfig)
}

// SubcommandFlags holds the flags required by the subcommands of
// `kubefed`.
type SubcommandFlags struct {
	Name                      string
	Host                      string
	FederationSystemNamespace string
	Kubeconfig                string
}

// AddSubcommandFlags adds the definition for `kubefed` subcommand
// flags.
func AddSubcommandFlags(cmd *cobra.Command) {
	cmd.Flags().String("kubeconfig", "", "Path to the kubeconfig file to use for CLI requests.")
	cmd.Flags().String("host-cluster-context", "", "Host cluster context")
	cmd.Flags().String("federation-system-namespace", DefaultFederationSystemNamespace, "Namespace in the host cluster where the federation system components are installed")
}

// GetSubcommandFlags retrieves the command line flag values for the
// `kubefed` subcommands.
func GetSubcommandFlags(cmd *cobra.Command, args []string) (*SubcommandFlags, error) {
	name, err := kubectlcmd.NameFromCommandArgs(cmd, args)
	if err != nil {
		return nil, err
	}
	return &SubcommandFlags{
		Name: name,
		Host: cmdutil.GetFlagString(cmd, "host-cluster-context"),
		FederationSystemNamespace: cmdutil.GetFlagString(cmd, "federation-system-namespace"),
		Kubeconfig:                cmdutil.GetFlagString(cmd, "kubeconfig"),
	}, nil
}

func CreateKubeconfigSecret(clientset *client.Clientset, kubeconfig *clientcmdapi.Config, namespace, name string, dryRun bool) (*api.Secret, error) {
	configBytes, err := clientcmd.Write(*kubeconfig)
	if err != nil {
		return nil, err
	}

	// Build the secret object with the minified and flattened
	// kubeconfig content.
	secret := &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Data: map[string][]byte{
			KubeconfigSecretDataKey: configBytes,
		},
	}

	if !dryRun {
		return clientset.Core().Secrets(namespace).Create(secret)
	}
	return secret, nil
}

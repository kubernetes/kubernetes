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

package kubefed

import (
	"fmt"
	"io"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/unversioned/clientcmd"
	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
	"k8s.io/kubernetes/pkg/kubectl"
	kubectlcmd "k8s.io/kubernetes/pkg/kubectl/cmd"
	"k8s.io/kubernetes/pkg/kubectl/cmd/templates"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/runtime"

	"github.com/spf13/cobra"
)

const (
	// KubeconfigSecretDataKey is the key name used in the secret to
	// stores a cluster's credentials.
	KubeconfigSecretDataKey = "kubeconfig"
)

var (
	join_long = templates.LongDesc(`
		Join a cluster to a federation.

        Current context is assumed to be a federation endpoint.
        Please use the --context flag otherwise.`)
	join_example = templates.Examples(`
		# Join a cluster to a federation by specifying the
		# cluster context name and the context name of the
		# federation control plane's host cluster.
		kubectl join foo --host=bar`)
)

// JoinFederationConfig provides a filesystem based kubeconfig (via
// `PathOptions()`) and a mechanism to talk to the federation host
// cluster.
type JoinFederationConfig interface {
	// PathOptions provides filesystem based kubeconfig access.
	PathOptions() *clientcmd.PathOptions
	// HostFactory provides a mechanism to communicate with the
	// cluster where federation control plane is hosted.
	HostFactory(host, kubeconfigPath string) cmdutil.Factory
}

// joinFederationConfig implements JoinFederationConfig interface.
type joinFederationConfig struct {
	pathOptions *clientcmd.PathOptions
}

func NewJoinFederationConfig(pathOptions *clientcmd.PathOptions) JoinFederationConfig {
	return &joinFederationConfig{
		pathOptions: pathOptions,
	}
}

func (r *joinFederationConfig) PathOptions() *clientcmd.PathOptions {
	return r.pathOptions
}

func (r *joinFederationConfig) HostFactory(host, kubeconfigPath string) cmdutil.Factory {
	loadingRules := *r.pathOptions.LoadingRules
	loadingRules.Precedence = r.pathOptions.GetLoadingPrecedence()
	loadingRules.ExplicitPath = kubeconfigPath
	overrides := &clientcmd.ConfigOverrides{
		CurrentContext: host,
	}

	hostClientConfig := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(&loadingRules, overrides)

	return cmdutil.NewFactory(hostClientConfig)
}

// NewCmdJoin defines the `join` command that joins a cluster to a
// federation.
func NewCmdJoin(f cmdutil.Factory, cmdOut io.Writer, config JoinFederationConfig) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "join CLUSTER_CONTEXT --host=HOST_CONTEXT",
		Short:   "Join a cluster to a federation",
		Long:    join_long,
		Example: join_example,
		Run: func(cmd *cobra.Command, args []string) {
			err := joinFederation(f, cmdOut, config, cmd, args)
			cmdutil.CheckErr(err)
		},
	}

	cmdutil.AddApplyAnnotationFlags(cmd)
	cmdutil.AddValidateFlags(cmd)
	cmdutil.AddPrinterFlags(cmd)
	cmdutil.AddGeneratorFlags(cmd, cmdutil.ClusterV1Beta1GeneratorName)
	addJoinFlags(cmd)
	return cmd
}

// joinFederation is the implementation of the `join federation` command.
func joinFederation(f cmdutil.Factory, cmdOut io.Writer, config JoinFederationConfig, cmd *cobra.Command, args []string) error {
	name, err := kubectlcmd.NameFromCommandArgs(cmd, args)
	if err != nil {
		return err
	}
	host := cmdutil.GetFlagString(cmd, "host")
	hostSystemNamespace := cmdutil.GetFlagString(cmd, "host-system-namespace")
	kubeconfig := cmdutil.GetFlagString(cmd, "kubeconfig")
	dryRun := cmdutil.GetDryRunFlag(cmd)

	po := config.PathOptions()
	po.LoadingRules.ExplicitPath = kubeconfig
	clientConfig, err := po.GetStartingConfig()
	if err != nil {
		return err
	}
	generator, err := clusterGenerator(clientConfig, name)
	if err != nil {
		return err
	}

	// We are not using the `kubectl create secret` machinery through
	// `RunCreateSubcommand` as we do to the cluster resource below
	// because we have a bunch of requirements that the machinery does
	// not satisfy.
	// 1. We want to create the secret in a specific namespace, which
	//    is neither the "default" namespace nor the one specified
	//    via the `--namespace` flag.
	// 2. `SecretGeneratorV1` requires LiteralSources in a string-ified
	//    form that it parses to generate the secret data key-value
	//    pairs. We, however, have the key-value pairs ready without a
	//    need for parsing.
	// 3. The result printing mechanism needs to be mostly quiet. We
	//    don't have to print the created secret in the default case.
	// Having said that, secret generation machinery could be altered to
	// suit our needs, but it is far less invasive and readable this way.
	hostFactory := config.HostFactory(host, kubeconfig)
	_, err = createSecret(hostFactory, clientConfig, hostSystemNamespace, name, dryRun)
	if err != nil {
		return err
	}

	return kubectlcmd.RunCreateSubcommand(f, cmd, cmdOut, &kubectlcmd.CreateSubcommandOptions{
		Name:                name,
		StructuredGenerator: generator,
		DryRun:              dryRun,
		OutputFormat:        cmdutil.GetFlagString(cmd, "output"),
	})
}

func addJoinFlags(cmd *cobra.Command) {
	cmd.Flags().String("kubeconfig", "", "Path to the kubeconfig file to use for CLI requests.")
	cmd.Flags().String("host", "", "Host cluster context")
	cmd.Flags().String("host-system-namespace", "federation-system", "Namespace in the host cluster where the federation system components are installed")
}

// minifyConfig is a wrapper around `clientcmdapi.MinifyConfig()` that
// sets the current context to the given context before calling
// `clientcmdapi.MinifyConfig()`.
func minifyConfig(clientConfig *clientcmdapi.Config, context string) (*clientcmdapi.Config, error) {
	// MinifyConfig inline-modifies the passed clientConfig. So we make a
	// copy of it before passing the config to it. A shallow copy is
	// sufficient because the underlying fields will be reconstructed by
	// MinifyConfig anyway.
	newClientConfig := *clientConfig
	newClientConfig.CurrentContext = context
	err := clientcmdapi.MinifyConfig(&newClientConfig)
	if err != nil {
		return nil, err
	}
	return &newClientConfig, nil
}

// createSecret extracts the kubeconfig for a given cluster and populates
// a secret with that kubeconfig.
func createSecret(hostFactory cmdutil.Factory, clientConfig *clientcmdapi.Config, namespace, name string, dryRun bool) (runtime.Object, error) {
	// Minify the kubeconfig to ensure that there is only information
	// relevant to the cluster we are registering.
	newClientConfig, err := minifyConfig(clientConfig, name)
	if err != nil {
		return nil, err
	}

	// Flatten the kubeconfig to ensure that all the referenced file
	// contents are inlined.
	err = clientcmdapi.FlattenConfig(newClientConfig)
	if err != nil {
		return nil, err
	}

	configBytes, err := clientcmd.Write(*newClientConfig)
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
		// Boilerplate to create the secret in the host cluster.
		clientset, err := hostFactory.ClientSet()
		if err != nil {
			return nil, err
		}
		return clientset.Core().Secrets(namespace).Create(secret)
	}
	return secret, nil
}

// clusterGenerator extracts the cluster information from the supplied
// kubeconfig and builds a StructuredGenerator for the
// `federation/cluster` API resource.
func clusterGenerator(clientConfig *clientcmdapi.Config, name string) (kubectl.StructuredGenerator, error) {
	// Get the context from the config.
	ctx, found := clientConfig.Contexts[name]
	if !found {
		return nil, fmt.Errorf("cluster context %q not found", name)
	}

	// Get the cluster object corresponding to the supplied context.
	cluster, found := clientConfig.Clusters[ctx.Cluster]
	if !found {
		return nil, fmt.Errorf("cluster endpoint not found for %q", name)
	}

	// Parse the cluster APIServer endpoint into a `urlScheme` struct
	// to extract and normalize the schema. If the schema is specified,
	// it is used as is. It is otherwise inferred from the port in the
	// host:port part of the endpoint URL.
	u, err := urlParse(cluster.Server)
	if err != nil {
		return nil, fmt.Errorf("failed to parse cluster endpoint %q for cluster %q", cluster.Server, name)
	}
	serverAddress := cluster.Server
	if u.scheme == "" {
		// Use "https" as the default scheme if the port isn't "80". It
		// is not a perfect inference, it is just a safe heuristic.
		scheme := "https"
		if u.port == "80" {
			scheme = "http"
		}
		serverAddress = strings.Join([]string{scheme, serverAddress}, "://")
	}

	generator := &kubectl.ClusterGeneratorV1Beta1{
		Name:          name,
		ServerAddress: serverAddress,
		SecretName:    name,
	}
	return generator, nil
}

// url represents a APIServer URL in the structured form.
type urlScheme struct {
	scheme string
	host   string
	port   string
	path   string
}

// urlParse implements a logic similar to `net/url.Parse()`, but unlike
// the standard library function, it gracefully handles the URLs without
// scheme. On the other hand, it only parses the segments we care about.
func urlParse(rawurl string) (*urlScheme, error) {
	scheme := ""
	port := ""
	path := ""
	hostport := rawurl
	segs := strings.SplitN(rawurl, "://", 2)
	if len(segs) == 2 {
		scheme = segs[0]
		hostport = segs[1]
	}

	remsegs := strings.SplitN(hostport, "/", 2)
	if len(remsegs) == 2 {
		hostport = remsegs[0]
		path = remsegs[1]
	}

	host := hostport
	hpsegs := strings.SplitN(hostport, ":", 2)
	if len(hpsegs) == 2 {
		host = hpsegs[0]
		port = hpsegs[1]
	}

	return &urlScheme{
		scheme: scheme,
		host:   host,
		port:   port,
		path:   path,
	}, nil
}

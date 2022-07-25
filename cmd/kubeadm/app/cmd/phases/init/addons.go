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

package phases

import (
	"io"

	"github.com/pkg/errors"
	"github.com/spf13/pflag"

	clientset "k8s.io/client-go/kubernetes"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	dnsaddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	proxyaddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
)

var (
	coreDNSAddonLongDesc = cmdutil.LongDesc(`
		Install the CoreDNS addon components via the API server.
		Please note that although the DNS server is deployed, it will not be scheduled until CNI is installed.
		`)

	kubeProxyAddonLongDesc = cmdutil.LongDesc(`
		Install the kube-proxy addon components via the API server.
		`)

	printManifest bool = false
)

// NewAddonPhase returns the addon Cobra command
func NewAddonPhase() workflow.Phase {
	dnsLocalFlags := pflag.NewFlagSet(options.PrintManifest, pflag.ContinueOnError)
	dnsLocalFlags.BoolVar(&printManifest, options.PrintManifest, printManifest, "Print the addon manifests to STDOUT instead of installing them")

	proxyLocalFlags := pflag.NewFlagSet(options.PrintManifest, pflag.ContinueOnError)
	proxyLocalFlags.BoolVar(&printManifest, options.PrintManifest, printManifest, "Print the addon manifests to STDOUT instead of installing them")

	return workflow.Phase{
		Name:  "addon",
		Short: "Install required addons for passing conformance tests",
		Long:  cmdutil.MacroCommandLongDescription,
		Phases: []workflow.Phase{
			{
				Name:           "all",
				Short:          "Install all the addons",
				InheritFlags:   getAddonPhaseFlags("all"),
				RunAllSiblings: true,
			},
			{
				Name:         "coredns",
				Short:        "Install the CoreDNS addon to a Kubernetes cluster",
				Long:         coreDNSAddonLongDesc,
				InheritFlags: getAddonPhaseFlags("coredns"),
				Run:          runCoreDNSAddon,
				LocalFlags:   dnsLocalFlags,
			},
			{
				Name:         "kube-proxy",
				Short:        "Install the kube-proxy addon to a Kubernetes cluster",
				Long:         kubeProxyAddonLongDesc,
				InheritFlags: getAddonPhaseFlags("kube-proxy"),
				Run:          runKubeProxyAddon,
				LocalFlags:   proxyLocalFlags,
			},
		},
	}
}

func getInitData(c workflow.RunData) (*kubeadmapi.InitConfiguration, clientset.Interface, io.Writer, error) {
	data, ok := c.(InitData)
	if !ok {
		return nil, nil, nil, errors.New("addon phase invoked with an invalid data struct")
	}
	cfg := data.Cfg()
	var client clientset.Interface
	var err error
	if !printManifest {
		client, err = data.Client()
		if err != nil {
			return nil, nil, nil, err
		}
	}

	out := data.OutputWriter()
	return cfg, client, out, err
}

// runCoreDNSAddon installs CoreDNS addon to a Kubernetes cluster
func runCoreDNSAddon(c workflow.RunData) error {
	cfg, client, out, err := getInitData(c)
	if err != nil {
		return err
	}
	return dnsaddon.EnsureDNSAddon(&cfg.ClusterConfiguration, client, out, printManifest)
}

// runKubeProxyAddon installs KubeProxy addon to a Kubernetes cluster
func runKubeProxyAddon(c workflow.RunData) error {
	cfg, client, out, err := getInitData(c)
	if err != nil {
		return err
	}
	return proxyaddon.EnsureProxyAddon(&cfg.ClusterConfiguration, &cfg.LocalAPIEndpoint, client, out, printManifest)
}

func getAddonPhaseFlags(name string) []string {
	flags := []string{
		options.CfgPath,
		options.KubeconfigPath,
		options.KubernetesVersion,
		options.ImageRepository,
	}
	if name == "all" || name == "kube-proxy" {
		flags = append(flags,
			options.APIServerAdvertiseAddress,
			options.ControlPlaneEndpoint,
			options.APIServerBindPort,
			options.NetworkingPodSubnet,
		)
	}
	if name == "all" || name == "coredns" {
		flags = append(flags,
			options.FeatureGatesString,
			options.NetworkingDNSDomain,
			options.NetworkingServiceSubnet,
		)
	}
	return flags
}

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
	"github.com/pkg/errors"

	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	dnsaddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	proxyaddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	coreDNSAddonLongDesc = normalizer.LongDesc(`
		Installs the CoreDNS addon components via the API server.
		Please note that although the DNS server is deployed, it will not be scheduled until CNI is installed.
		`)

	kubeProxyAddonLongDesc = normalizer.LongDesc(`
		Installs the kube-proxy addon components via the API server.
		`)
)

type addonData interface {
	Cfg() *kubeadmapi.InitConfiguration
	Client() (clientset.Interface, error)
}

// NewAddonPhase returns the addon Cobra command
func NewAddonPhase() workflow.Phase {
	return workflow.Phase{
		Name:  "addon",
		Short: "Installs required addons for passing Conformance tests",
		Long:  cmdutil.MacroCommandLongDescription,
		Phases: []workflow.Phase{
			{
				Name:           "all",
				Short:          "Installs all the addons",
				InheritFlags:   getAddonPhaseFlags("all"),
				RunAllSiblings: true,
			},
			{
				Name:         "coredns",
				Short:        "Installs the CoreDNS addon to a Kubernetes cluster",
				Long:         coreDNSAddonLongDesc,
				InheritFlags: getAddonPhaseFlags("coredns"),
				Run:          runCoreDNSAddon,
			},
			{
				Name:         "kube-proxy",
				Short:        "Installs the kube-proxy addon to a Kubernetes cluster",
				Long:         kubeProxyAddonLongDesc,
				InheritFlags: getAddonPhaseFlags("kube-proxy"),
				Run:          runKubeProxyAddon,
			},
		},
	}
}

func getAddonData(c workflow.RunData) (*kubeadmapi.InitConfiguration, clientset.Interface, error) {
	data, ok := c.(addonData)
	if !ok {
		return nil, nil, errors.New("addon phase invoked with an invalid data struct")
	}
	cfg := data.Cfg()
	client, err := data.Client()
	if err != nil {
		return nil, nil, err
	}
	return cfg, client, err
}

// runCoreDNSAddon installs CoreDNS addon to a Kubernetes cluster
func runCoreDNSAddon(c workflow.RunData) error {
	cfg, client, err := getAddonData(c)
	if err != nil {
		return err
	}
	return dnsaddon.EnsureDNSAddon(cfg, client)
}

// runKubeProxyAddon installs KubeProxy addon to a Kubernetes cluster
func runKubeProxyAddon(c workflow.RunData) error {
	cfg, client, err := getAddonData(c)
	if err != nil {
		return err
	}
	return proxyaddon.EnsureProxyAddon(cfg, client)
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

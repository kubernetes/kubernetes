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
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	dnsaddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	installeraddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/installer"
	proxyaddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"
)

var (
	coreDNSAddonLongDesc = cmdutil.LongDesc(`
		Install the CoreDNS addon components via the API server.
		Please note that although the DNS server is deployed, it will not be scheduled until CNI is installed.
		This phase does not run when ClusterConfiguration.FeatureGates.AddonInstaller is true.
		`)

	kubeProxyAddonLongDesc = cmdutil.LongDesc(`
		Install the kube-proxy addon components via the API server.
		This phase does not run when ClusterConfiguration.FeatureGates.AddonInstaller is true.
		`)

	addonInstallerLongDesc = cmdutil.LongDesc(`
		Install addons from an AddonInstallerConfiguration via the API server.
		This phase only runs when ClusterConfiguration.FeatureGates.AddonInstaller is true.
		`)  // TODO: add documentation / link
)

// NewAddonPhase returns the addon Cobra command
func NewAddonPhase() workflow.Phase {
	return workflow.Phase{
		Name:  "addon",
		Short: "Install required addons for passing Conformance tests",
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
			},
			{
				Name:         "kube-proxy",
				Short:        "Install the kube-proxy addon to a Kubernetes cluster",
				Long:         kubeProxyAddonLongDesc,
				InheritFlags: getAddonPhaseFlags("kube-proxy"),
				Run:          runKubeProxyAddon,
			},
			{
				Name:         "installer",
				Short:        "Install addons from an AddonInstallerConfiguration",
				Long:         addonInstallerLongDesc,
				InheritFlags: getAddonPhaseFlags("installer"),
				Run:          runAddonInstaller,
			},
		},
	}
}

func getInitData(c workflow.RunData) (InitData, *kubeadmapi.InitConfiguration, clientset.Interface, error) {
	data, ok := c.(InitData)
	if !ok {
		return data, nil, nil, errors.New("addon phase invoked with an invalid data struct")
	}
	cfg := data.Cfg()
	client, err := data.Client()
	if err != nil {
		return data, nil, nil, err
	}
	return data, cfg, client, err
}

// runCoreDNSAddon installs CoreDNS addon to a Kubernetes cluster
func runCoreDNSAddon(c workflow.RunData) error {
	_, cfg, client, err := getInitData(c)
	if err != nil {
		return err
	}
	if features.Enabled(cfg.ClusterConfiguration.FeatureGates, features.AddonInstaller) {
		return nil
	}
	return dnsaddon.EnsureDNSAddon(&cfg.ClusterConfiguration, client)
}

// runKubeProxyAddon installs KubeProxy addon to a Kubernetes cluster
func runKubeProxyAddon(c workflow.RunData) error {
	_, cfg, client, err := getInitData(c)
	if err != nil {
		return err
	}
	if features.Enabled(cfg.ClusterConfiguration.FeatureGates, features.AddonInstaller) {
		return nil
	}
	return proxyaddon.EnsureProxyAddon(&cfg.ClusterConfiguration, &cfg.LocalAPIEndpoint, client)
}

// runAddonInstaller executes the addon-installer to apply the default or user-configured addons
func runAddonInstaller(c workflow.RunData) error {
	initData, cfg, _, err := getInitData(c)
	if err != nil {
		return err
	}
	if features.Enabled(cfg.ClusterConfiguration.FeatureGates, features.AddonInstaller) {
		return installeraddon.ApplyAddonConfiguration(cfg, initData.DryRun(), initData.RealKubeConfigPath())
	}
	return nil
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
			options.NetworkingDNSDomain,
			options.NetworkingServiceSubnet,
		)
	}
	if name == "all" || name == "coredns" || name == "installer" {
		flags = append(flags,
			options.FeatureGatesString,
		)
	}
	if name == "all" || name == "installer" {
		flags = append(flags,
			options.DryRun,
		)
	}
	return flags
}

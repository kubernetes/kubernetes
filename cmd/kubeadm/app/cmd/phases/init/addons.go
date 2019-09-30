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
	"os"

	"github.com/pkg/errors"

	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	dnsaddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/dns"
	proxyaddon "k8s.io/kubernetes/cmd/kubeadm/app/phases/addons/proxy"

	addoninstall "sigs.k8s.io/addon-operators/installer/install"
)

var (
	coreDNSAddonLongDesc = cmdutil.LongDesc(`
		Install the CoreDNS addon components via the API server.
		Please note that although the DNS server is deployed, it will not be scheduled until CNI is installed.
		`)

	kubeProxyAddonLongDesc = cmdutil.LongDesc(`
		Install the kube-proxy addon components via the API server.
		`)

	addonInstallerLongDesc = cmdutil.LongDesc(`
		Install addons from an AddonInstallerConfiguration via the API server.
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

func getInitData(c workflow.RunData) (*kubeadmapi.InitConfiguration, clientset.Interface, error) {
	data, ok := c.(InitData)
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
	cfg, client, err := getInitData(c)
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
	cfg, client, err := getInitData(c)
	if err != nil {
		return err
	}
	if features.Enabled(cfg.ClusterConfiguration.FeatureGates, features.AddonInstaller) {
		return nil
	}
	return proxyaddon.EnsureProxyAddon(&cfg.ClusterConfiguration, &cfg.LocalAPIEndpoint, client)
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
	if name == "all" || name == "installer" {
		// TODO
		// flags = append(flags,
		// 	options.AddonInstallerDryRun
		// )
	}
	return flags
}

// runAddonInstaller executes the addon-installer
func runAddonInstaller(c workflow.RunData) error {
	// cfg, client, err := getInitData(c)
	cfg, _, err := getInitData(c)
	// TODO: getAddonInstallerFromConfigMap
	if err != nil {
		return err
	}
	if features.Enabled(cfg.ClusterConfiguration.FeatureGates, features.AddonInstaller) {
		installCfg := cfg.ClusterConfiguration.ComponentConfigs.AddonInstaller
		if installCfg == nil {
			return errors.New("addoninstaller phase invoked with nil AddonInstaller ComponentConfig")
		}
		r := addoninstall.Runtime{
			Config: installCfg,
			Stdout: os.Stdout,
			Stderr: os.Stderr,
		}
		// sigs := make(chan os.Signal, 1)
		// signal.Notify(sigs, syscall.SIGINT, syscall.SIGTERM)
		// go func() {
		// 	errs := r.HandleSignal(<-sigs)
		// 	for _, err := range errs {
		// 		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		// 	}
		// }()

		err = r.CheckDeps()
		if err != nil {
			return err
		}
		err = r.CheckConfig()
		if err != nil {
			return err
		}
		err = r.InstallAddons()
		if err != nil {
			return err
		}
	}
	return nil
}

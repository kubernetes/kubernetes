/*
Copyright 2017 The Kubernetes Authors.

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

package upgrade

import (
	"bufio"
	"bytes"
	"io"
	"os"
	"path/filepath"

	"github.com/spf13/pflag"

	"k8s.io/apimachinery/pkg/util/sets"
	fakediscovery "k8s.io/client-go/discovery/fake"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
	staticpodutil "k8s.io/kubernetes/cmd/kubeadm/app/util/staticpod"
)

// enforceRequirements verifies that it's okay to upgrade and then returns the variables needed for the rest of the procedure.
// It is only used by `kubeadm upgrade plan`, which never applies changes and therefore never dry-runs.
func enforceRequirements(flagSet *pflag.FlagSet, flags *applyPlanFlags, args []string, printer output.Printer) (clientset.Interface, upgrade.VersionGetter, *kubeadmapi.InitConfiguration, *kubeadmapi.UpgradeConfiguration, error) {
	externalCfg := &kubeadmapiv1.UpgradeConfiguration{}
	opt := configutil.LoadOrDefaultConfigurationOptions{}
	upgradeCfg, err := configutil.LoadOrDefaultUpgradeConfiguration(flags.cfgPath, externalCfg, opt)
	if err != nil {
		return nil, nil, nil, nil, errors.Wrap(err, "[upgrade/upgrade config] FATAL")
	}

	const isDryRun = false

	client, err := getClient(flags.kubeConfigPath, isDryRun, printer)
	if err != nil {
		return nil, nil, nil, nil, errors.Wrapf(err, "couldn't create a Kubernetes client from file %q", flags.kubeConfigPath)
	}

	// The union of pre-flight errors is not stored back into .Plan.IgnorePreflightErrors as it's not used.
	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(flags.ignorePreflightErrors, upgradeCfg.Plan.IgnorePreflightErrors)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	// Ensure the user is root
	klog.V(1).Info("running preflight checks")
	if err := runPreflightChecks(client, ignorePreflightErrorsSet, printer); err != nil {
		return nil, nil, nil, nil, err
	}

	getNodeRegistration := true
	getAPIEndpoint := staticpodutil.IsControlPlaneNode()
	getComponentConfigs := true
	initCfg, err := configutil.FetchInitConfigurationFromCluster(client, printer, "upgrade/config", getNodeRegistration, getAPIEndpoint, getComponentConfigs, false)
	if err != nil {
		return nil, nil, nil, nil, errors.Wrap(err, "[upgrade/init config] FATAL")
	}

	newK8sVersion := upgradeCfg.Plan.KubernetesVersion
	// If option was specified in both args and config file, args will overwrite the config file.
	if len(args) == 1 {
		newK8sVersion = args[0]
	}

	// Run healthchecks against the cluster
	if err := upgrade.CheckClusterHealth(client, &initCfg.ClusterConfiguration, ignorePreflightErrorsSet, isDryRun, printer); err != nil {
		return nil, nil, nil, nil, errors.Wrap(err, "[upgrade/health] FATAL")
	}

	// Check if feature gate flags used in the cluster are consistent with the set of features currently supported by kubeadm
	if msg := features.CheckDeprecatedFlags(&features.InitFeatureGates, initCfg.FeatureGates); len(msg) > 0 {
		for _, m := range msg {
			printer.Printf("[upgrade/config] %s\n", m)
		}
	}

	// If the user told us to print this information out; do it!
	printConfig, ok := cmdutil.ValueFromFlagsOrConfig(flagSet, options.PrintConfig, upgradeCfg.Plan.PrintConfig, &flags.printConfig).(*bool)
	if ok && *printConfig {
		printConfiguration(&initCfg.ClusterConfiguration, os.Stdout, printer)
	} else if !ok {
		return nil, nil, nil, nil, cmdutil.TypeMismatchErr("printConfig", "bool")
	}

	// Use a real version getter interface that queries the API server, the kubeadm client and the Kubernetes CI system for latest versions
	return client, upgrade.NewOfflineVersionGetter(upgrade.NewKubeVersionGetter(client), newK8sVersion), initCfg, upgradeCfg, nil
}

// printConfiguration prints the external version of the API to yaml
func printConfiguration(clustercfg *kubeadmapi.ClusterConfiguration, w io.Writer, printer output.Printer) {
	// Short-circuit if cfg is nil, so we can safely get the value of the pointer below
	if clustercfg == nil {
		return
	}

	cfgYaml, err := configutil.MarshalKubeadmConfigObject(clustercfg, kubeadmapiv1.SchemeGroupVersion)
	if err == nil {
		printer.Fprintln(w, "[upgrade/config] Configuration used:")

		scanner := bufio.NewScanner(bytes.NewReader(cfgYaml))
		for scanner.Scan() {
			printer.Fprintf(w, "\t%s\n", scanner.Text())
		}
	}
}

// runPreflightChecks runs the root preflight check
func runPreflightChecks(client clientset.Interface, ignorePreflightErrors sets.Set[string], printer output.Printer) error {
	printer.Printf("[preflight] Running pre-flight checks.\n")
	err := preflight.RunRootCheckOnly(ignorePreflightErrors)
	if err != nil {
		return err
	}
	return upgrade.RunCoreDNSMigrationCheck(client, ignorePreflightErrors)
}

// getClient gets a real or fake client depending on whether the user is dry-running or not
func getClient(file string, dryRun bool, printer output.Printer) (clientset.Interface, error) {
	if dryRun {
		// Default the server version to the kubeadm version.
		serverVersion := constants.CurrentKubernetesVersion.Info()

		dryRun := apiclient.NewDryRun()
		dryRun.WithDefaultMarshalFunction().
			WithWriter(os.Stdout).
			PrependReactor(dryRun.HealthCheckJobReactor()).
			PrependReactor(dryRun.PatchNodeReactor())

		// If the kubeconfig exists, construct a real client from it and get the real serverVersion.
		if _, err := os.Stat(file); err == nil {
			_, _ = printer.Printf("[dryrun] Creating a real client from %q\n", file)
			if err := dryRun.WithKubeConfigFile(file); err != nil {
				return nil, err
			}
			serverVersion, err = dryRun.Client().Discovery().ServerVersion()
			if err != nil {
				return nil, errors.Wrap(err, "failed to get server version")
			}
		} else if os.IsNotExist(err) {
			// If the file (supposedly admin.conf) does not exist, add more reactors.
			// Knowing the node name is required by the ListPodsReactor. For that we try to use
			// the kubelet.conf client, if it exists. If not, it falls back to hostname.
			_, _ = printer.Printf("[dryrun] Dryrunning without a real client\n")
			kubeconfigPath := filepath.Join(constants.KubernetesDir, constants.KubeletKubeConfigFileName)
			nodeName, err := configutil.GetNodeName(kubeconfigPath)
			if err != nil {
				return nil, err
			}
			dryRun.PrependReactor(dryRun.GetKubeadmConfigReactor()).
				PrependReactor(dryRun.GetKubeletConfigReactor()).
				PrependReactor(dryRun.GetKubeProxyConfigReactor()).
				PrependReactor(dryRun.GetNodeReactor()).
				PrependReactor(dryRun.ListPodsReactor(nodeName)).
				PrependReactor(dryRun.GetCoreDNSConfigReactor()).
				PrependReactor(dryRun.ListDeploymentsReactor())
		} else {
			// Throw an error if the file exists but there was a different stat error.
			return nil, errors.Wrapf(err, "could not create a client from %q", file)
		}

		// Obtain the FakeDiscovery object for this fake client.
		fakeClient := dryRun.FakeClient()
		fakeClientDiscovery, ok := fakeClient.Discovery().(*fakediscovery.FakeDiscovery)
		if !ok {
			return nil, errors.New("could not set fake discovery's server version")
		}
		// Set the right server version for it.
		fakeClientDiscovery.FakedServerVersion = serverVersion

		return fakeClient, nil
	}
	return kubeconfigutil.ClientSetFromFile(file)
}

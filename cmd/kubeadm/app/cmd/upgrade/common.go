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

	"github.com/pkg/errors"
	"github.com/spf13/pflag"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	fakediscovery "k8s.io/client-go/discovery/fake"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"

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
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

// enforceRequirements verifies that it's okay to upgrade and then returns the variables needed for the rest of the procedure
func enforceRequirements(flagSet *pflag.FlagSet, flags *applyPlanFlags, args []string, dryRun bool, upgradeApply bool, printer output.Printer) (clientset.Interface, upgrade.VersionGetter, *kubeadmapi.InitConfiguration, *kubeadmapi.UpgradeConfiguration, error) {
	externalCfg := &kubeadmapiv1.UpgradeConfiguration{}
	opt := configutil.LoadOrDefaultConfigurationOptions{}
	upgradeCfg, err := configutil.LoadOrDefaultUpgradeConfiguration(flags.cfgPath, externalCfg, opt)
	if err != nil {
		return nil, nil, nil, nil, errors.Wrap(err, "[upgrade/upgrade config] FATAL")
	}

	// `dryRun` should be always be `false` for `kubeadm plan`.
	isDryRun := ptr.To(false)
	printConfigCfg := upgradeCfg.Plan.PrintConfig
	ignoreErrCfg := upgradeCfg.Plan.IgnorePreflightErrors
	ok := false
	if upgradeApply {
		printConfigCfg = upgradeCfg.Apply.PrintConfig
		ignoreErrCfg = upgradeCfg.Apply.IgnorePreflightErrors
		isDryRun, ok = cmdutil.ValueFromFlagsOrConfig(flagSet, options.DryRun, upgradeCfg.Apply.DryRun, &dryRun).(*bool)
		if !ok {
			return nil, nil, nil, nil, cmdutil.TypeMismatchErr("dryRun", "bool")
		}
	}

	client, err := getClient(flags.kubeConfigPath, *isDryRun)
	if err != nil {
		return nil, nil, nil, nil, errors.Wrapf(err, "couldn't create a Kubernetes client from file %q", flags.kubeConfigPath)
	}

	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(flags.ignorePreflightErrors, ignoreErrCfg)
	if err != nil {
		return nil, nil, nil, nil, err
	}

	// Also set the union of pre-flight errors to UpgradeConfiguration, to provide a consistent view of the runtime configuration.
	// .Plan.IgnorePreflightErrors is not set as it's not used.
	if upgradeApply {
		upgradeCfg.Apply.IgnorePreflightErrors = sets.List(ignorePreflightErrorsSet)
	}

	// Ensure the user is root
	klog.V(1).Info("running preflight checks")
	if err := runPreflightChecks(client, ignorePreflightErrorsSet, printer); err != nil {
		return nil, nil, nil, nil, err
	}

	initCfg, err := configutil.FetchInitConfigurationFromCluster(client, printer, "upgrade/config", false, false)
	if err != nil {
		if apierrors.IsNotFound(err) {
			_, _ = printer.Printf("[upgrade/config] In order to upgrade, a ConfigMap called %q in the %q namespace must exist.\n", constants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
			_, _ = printer.Printf("[upgrade/config] Use 'kubeadm init phase upload-config --config your-config.yaml' to re-upload it.\n")
			err = errors.Errorf("the ConfigMap %q in the %q namespace was not found", constants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
		}
		return nil, nil, nil, nil, errors.Wrap(err, "[upgrade/init config] FATAL")
	}

	// Set the ImagePullPolicy and ImagePullSerial from the UpgradeApplyConfiguration to the InitConfiguration.
	// These are used by preflight.RunPullImagesCheck() when running 'apply'.
	if upgradeApply {
		initCfg.NodeRegistration.ImagePullPolicy = upgradeCfg.Apply.ImagePullPolicy
		initCfg.NodeRegistration.ImagePullSerial = upgradeCfg.Apply.ImagePullSerial
	}

	newK8sVersion := upgradeCfg.Plan.KubernetesVersion
	if upgradeApply {
		newK8sVersion = upgradeCfg.Apply.KubernetesVersion
		// The version arg is mandatory, during upgrade apply, unless it's specified in the config file
		if newK8sVersion == "" {
			if err := cmdutil.ValidateExactArgNumber(args, []string{"version"}); err != nil {
				return nil, nil, nil, nil, err
			}
		}
	}

	// If option was specified in both args and config file, args will overwrite the config file.
	if len(args) == 1 {
		newK8sVersion = args[0]
	}

	if upgradeApply {
		// The `upgrade apply` version always overwrites the KubernetesVersion in the returned cfg with the target
		// version. While this is not the same for `upgrade plan` where the KubernetesVersion should be the old
		// one (because the call to getComponentConfigVersionStates requires the currently installed version).
		// This also makes the KubernetesVersion value returned for `upgrade plan` consistent as that command
		// allows to not specify a target version in which case KubernetesVersion will always hold the currently
		// installed one.
		initCfg.KubernetesVersion = newK8sVersion
	}

	// Run healthchecks against the cluster
	if err := upgrade.CheckClusterHealth(client, &initCfg.ClusterConfiguration, ignorePreflightErrorsSet, printer); err != nil {
		return nil, nil, nil, nil, errors.Wrap(err, "[upgrade/health] FATAL")
	}

	// Check if feature gate flags used in the cluster are consistent with the set of features currently supported by kubeadm
	if msg := features.CheckDeprecatedFlags(&features.InitFeatureGates, initCfg.FeatureGates); len(msg) > 0 {
		for _, m := range msg {
			printer.Printf("[upgrade/config] %s\n", m)
		}
	}

	// If the user told us to print this information out; do it!
	printConfig, ok := cmdutil.ValueFromFlagsOrConfig(flagSet, options.PrintConfig, printConfigCfg, &flags.printConfig).(*bool)
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
func getClient(file string, dryRun bool) (clientset.Interface, error) {
	if dryRun {
		dryRunGetter, err := apiclient.NewClientBackedDryRunGetterFromKubeconfig(file)
		if err != nil {
			return nil, err
		}

		// In order for fakeclient.Discovery().ServerVersion() to return the backing API Server's
		// real version; we have to do some clever API machinery tricks. First, we get the real
		// API Server's version
		realServerVersion, err := dryRunGetter.Client().Discovery().ServerVersion()
		if err != nil {
			return nil, errors.Wrap(err, "failed to get server version")
		}

		// Get the fake clientset
		dryRunOpts := apiclient.GetDefaultDryRunClientOptions(dryRunGetter, os.Stdout)
		// Print GET and LIST requests
		dryRunOpts.PrintGETAndLIST = true
		fakeclient := apiclient.NewDryRunClientWithOpts(dryRunOpts)
		// As we know the return of Discovery() of the fake clientset is of type *fakediscovery.FakeDiscovery
		// we can convert it to that struct.
		fakeclientDiscovery, ok := fakeclient.Discovery().(*fakediscovery.FakeDiscovery)
		if !ok {
			return nil, errors.New("couldn't set fake discovery's server version")
		}
		// Lastly, set the right server version to be used
		fakeclientDiscovery.FakedServerVersion = realServerVersion
		// return the fake clientset used for dry-running
		return fakeclient, nil
	}
	return kubeconfigutil.ClientSetFromFile(file)
}

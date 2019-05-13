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
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/pkg/errors"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	fakediscovery "k8s.io/client-go/discovery/fake"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

func getK8sVersionFromUserInput(flags *applyPlanFlags, args []string, versionIsMandatory bool) (string, error) {
	var userVersion string

	// If the version is specified in config file, pick up that value.
	if flags.cfgPath != "" {
		// Note that cfg isn't preserved here, it's just an one-off to populate userVersion based on --config
		cfg, err := configutil.LoadInitConfigurationFromFile(flags.cfgPath)
		if err != nil {
			return "", err
		}

		userVersion = cfg.KubernetesVersion
	}

	// the version arg is mandatory unless version is specified in the config file
	if versionIsMandatory && userVersion == "" {
		if err := cmdutil.ValidateExactArgNumber(args, []string{"version"}); err != nil {
			return "", err
		}
	}

	// If option was specified in both args and config file, args will overwrite the config file.
	if len(args) == 1 {
		userVersion = args[0]
	}

	return userVersion, nil
}

// enforceRequirements verifies that it's okay to upgrade and then returns the variables needed for the rest of the procedure
func enforceRequirements(flags *applyPlanFlags, dryRun bool, newK8sVersion string) (clientset.Interface, upgrade.VersionGetter, *kubeadmapi.InitConfiguration, error) {
	ignorePreflightErrorsSet, err := validation.ValidateIgnorePreflightErrors(flags.ignorePreflightErrors)
	if err != nil {
		return nil, nil, nil, err
	}

	// Ensure the user is root
	klog.V(1).Info("running preflight checks")
	if err := runPreflightChecks(ignorePreflightErrorsSet); err != nil {
		return nil, nil, nil, err
	}

	client, err := getClient(flags.kubeConfigPath, dryRun)
	if err != nil {
		return nil, nil, nil, errors.Wrapf(err, "couldn't create a Kubernetes client from file %q", flags.kubeConfigPath)
	}

	// Check if the cluster is self-hosted
	if upgrade.IsControlPlaneSelfHosted(client) {
		return nil, nil, nil, errors.New("cannot upgrade a self-hosted control plane")
	}

	// Run healthchecks against the cluster
	if err := upgrade.CheckClusterHealth(client, ignorePreflightErrorsSet); err != nil {
		return nil, nil, nil, errors.Wrap(err, "[upgrade/health] FATAL")
	}

	// Fetch the configuration from a file or ConfigMap and validate it
	fmt.Println("[upgrade/config] Making sure the configuration is correct:")

	var cfg *kubeadmapi.InitConfiguration
	if flags.cfgPath != "" {
		cfg, err = configutil.LoadInitConfigurationFromFile(flags.cfgPath)
	} else {
		cfg, err = configutil.FetchInitConfigurationFromCluster(client, os.Stdout, "upgrade/config", false)
	}

	if err != nil {
		if apierrors.IsNotFound(err) {
			fmt.Printf("[upgrade/config] In order to upgrade, a ConfigMap called %q in the %s namespace must exist.\n", constants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
			fmt.Println("[upgrade/config] Without this information, 'kubeadm upgrade' won't know how to configure your upgraded cluster.")
			fmt.Println("")
			fmt.Println("[upgrade/config] Next steps:")
			fmt.Printf("\t- OPTION 1: Run 'kubeadm config upload from-flags' and specify the same CLI arguments you passed to 'kubeadm init' when you created your control-plane.\n")
			fmt.Printf("\t- OPTION 2: Run 'kubeadm config upload from-file' and specify the same config file you passed to 'kubeadm init' when you created your control-plane.\n")
			fmt.Printf("\t- OPTION 3: Pass a config file to 'kubeadm upgrade' using the --config flag.\n")
			fmt.Println("")
			err = errors.Errorf("the ConfigMap %q in the %s namespace used for getting configuration information was not found", constants.KubeadmConfigConfigMap, metav1.NamespaceSystem)
		}
		return nil, nil, nil, errors.Wrap(err, "[upgrade/config] FATAL")
	}

	// If a new k8s version should be set, apply the change before printing the config
	if len(newK8sVersion) != 0 {
		cfg.KubernetesVersion = newK8sVersion
	}

	// If features gates are passed to the command line, use it (otherwise use featureGates from configuration)
	if flags.featureGatesString != "" {
		cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, flags.featureGatesString)
		if err != nil {
			return nil, nil, nil, errors.Wrap(err, "[upgrade/config] FATAL")
		}
	}

	// Check if feature gate flags used in the cluster are consistent with the set of features currently supported by kubeadm
	if msg := features.CheckDeprecatedFlags(&features.InitFeatureGates, cfg.FeatureGates); len(msg) > 0 {
		for _, m := range msg {
			fmt.Printf("[upgrade/config] %s\n", m)
		}
		return nil, nil, nil, errors.New("[upgrade/config] FATAL. Unable to upgrade a cluster using deprecated feature-gate flags. Please see the release notes")
	}

	// If the user told us to print this information out; do it!
	if flags.printConfig {
		printConfiguration(&cfg.ClusterConfiguration, os.Stdout)
	}

	// Use a real version getter interface that queries the API server, the kubeadm client and the Kubernetes CI system for latest versions
	return client, upgrade.NewOfflineVersionGetter(upgrade.NewKubeVersionGetter(client, os.Stdout), newK8sVersion), cfg, nil
}

// printConfiguration prints the external version of the API to yaml
func printConfiguration(clustercfg *kubeadmapi.ClusterConfiguration, w io.Writer) {
	// Short-circuit if cfg is nil, so we can safely get the value of the pointer below
	if clustercfg == nil {
		return
	}

	cfgYaml, err := configutil.MarshalKubeadmConfigObject(clustercfg)
	if err == nil {
		fmt.Fprintln(w, "[upgrade/config] Configuration used:")

		scanner := bufio.NewScanner(bytes.NewReader(cfgYaml))
		for scanner.Scan() {
			fmt.Fprintf(w, "\t%s\n", scanner.Text())
		}
	}
}

// runPreflightChecks runs the root preflight check
func runPreflightChecks(ignorePreflightErrors sets.String) error {
	fmt.Println("[preflight] Running pre-flight checks.")
	return preflight.RunRootCheckOnly(ignorePreflightErrors)
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

// getWaiter gets the right waiter implementation
func getWaiter(dryRun bool, client clientset.Interface) apiclient.Waiter {
	if dryRun {
		return dryrunutil.NewWaiter()
	}
	return apiclient.NewKubeWaiter(client, upgrade.UpgradeManifestTimeout, os.Stdout)
}

// InteractivelyConfirmUpgrade asks the user whether they _really_ want to upgrade.
func InteractivelyConfirmUpgrade(question string) error {

	fmt.Printf("[upgrade/confirm] %s [y/N]: ", question)

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	if err := scanner.Err(); err != nil {
		return errors.Wrap(err, "couldn't read from standard input")
	}
	answer := scanner.Text()
	if strings.ToLower(answer) == "y" || strings.ToLower(answer) == "yes" {
		return nil
	}

	return errors.New("won't proceed; the user didn't answer (Y|y) in order to continue")
}

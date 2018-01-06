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

	"github.com/ghodss/yaml"

	"k8s.io/apimachinery/pkg/util/sets"
	fakediscovery "k8s.io/client-go/discovery/fake"
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	dryrunutil "k8s.io/kubernetes/cmd/kubeadm/app/util/dryrun"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// upgradeVariables holds variables needed for performing an upgrade or planning to do so
// TODO - Restructure or rename upgradeVariables
type upgradeVariables struct {
	client        clientset.Interface
	cfg           *kubeadmapiext.MasterConfiguration
	versionGetter upgrade.VersionGetter
	waiter        apiclient.Waiter
}

// enforceRequirements verifies that it's okay to upgrade and then returns the variables needed for the rest of the procedure
func enforceRequirements(flags *cmdUpgradeFlags, dryRun bool, newK8sVersion string) (*upgradeVariables, error) {
	client, err := getClient(flags.kubeConfigPath, dryRun)
	if err != nil {
		return nil, fmt.Errorf("couldn't create a Kubernetes client from file %q: %v", flags.kubeConfigPath, err)
	}

	// Run healthchecks against the cluster
	if err := upgrade.CheckClusterHealth(client, flags.ignorePreflightErrorsSet); err != nil {
		return nil, fmt.Errorf("[upgrade/health] FATAL: %v", err)
	}

	// Fetch the configuration from a file or ConfigMap and validate it
	cfg, err := upgrade.FetchConfiguration(client, os.Stdout, flags.cfgPath)
	if err != nil {
		return nil, fmt.Errorf("[upgrade/config] FATAL: %v", err)
	}

	// If a new k8s version should be set, apply the change before printing the config
	if len(newK8sVersion) != 0 {
		cfg.KubernetesVersion = newK8sVersion
	}

	// If the user told us to print this information out; do it!
	if flags.printConfig {
		printConfiguration(cfg, os.Stdout)
	}

	cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, flags.featureGatesString)
	if err != nil {
		return nil, fmt.Errorf("[upgrade/config] FATAL: %v", err)
	}

	return &upgradeVariables{
		client: client,
		cfg:    cfg,
		// Use a real version getter interface that queries the API server, the kubeadm client and the Kubernetes CI system for latest versions
		versionGetter: upgrade.NewKubeVersionGetter(client, os.Stdout),
		// Use the waiter conditionally based on the dryrunning variable
		waiter: getWaiter(dryRun, client),
	}, nil
}

// printConfiguration prints the external version of the API to yaml
func printConfiguration(cfg *kubeadmapiext.MasterConfiguration, w io.Writer) {
	// Short-circuit if cfg is nil, so we can safely get the value of the pointer below
	if cfg == nil {
		return
	}

	cfgYaml, err := yaml.Marshal(*cfg)
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
			return nil, fmt.Errorf("failed to get server version: %v", err)
		}

		// Get the fake clientset
		fakeclient := apiclient.NewDryRunClient(dryRunGetter, os.Stdout)
		// As we know the return of Discovery() of the fake clientset is of type *fakediscovery.FakeDiscovery
		// we can convert it to that struct.
		fakeclientDiscovery, ok := fakeclient.Discovery().(*fakediscovery.FakeDiscovery)
		if !ok {
			return nil, fmt.Errorf("couldn't set fake discovery's server version")
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
	return apiclient.NewKubeWaiter(client, upgradeManifestTimeout, os.Stdout)
}

// InteractivelyConfirmUpgrade asks the user whether they _really_ want to upgrade.
func InteractivelyConfirmUpgrade(question string) error {

	fmt.Printf("[upgrade/confirm] %s [y/N]: ", question)

	scanner := bufio.NewScanner(os.Stdin)
	scanner.Scan()
	if err := scanner.Err(); err != nil {
		return fmt.Errorf("couldn't read from standard input: %v", err)
	}
	answer := scanner.Text()
	if strings.ToLower(answer) == "y" || strings.ToLower(answer) == "yes" {
		return nil
	}

	return fmt.Errorf("won't proceed; the user didn't answer (Y|y) in order to continue")
}

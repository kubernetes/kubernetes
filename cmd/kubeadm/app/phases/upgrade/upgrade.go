/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"os"

	"github.com/pkg/errors"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	fakediscovery "k8s.io/client-go/discovery/fake"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
)

// GetUpgradeVariables is a function
func GetUpgradeVariables(kubeConfigPath string, cfgPath string, dryRun bool, newK8sVersion string) (clientset.Interface, *kubeadmapi.InitConfiguration, VersionGetter, error) {
	client, err := GetClient(kubeConfigPath, dryRun)
	if err != nil {
		return nil, nil, nil, errors.Wrapf(err, "couldn't create a Kubernetes client from file %q", kubeConfigPath)
	}

	var cfg *kubeadmapi.InitConfiguration
	if cfgPath != "" {
		klog.Warning("WARNING: Usage of the --config flag for reconfiguring the cluster during upgrade is not recommended!")
		cfg, err = configutil.LoadInitConfigurationFromFile(cfgPath)
	} else {
		cfg, err = configutil.FetchInitConfigurationFromCluster(client, os.Stdout, "upgrade/config", false)
	}

	// If a new k8s version should be set, apply the change before printing the config
	if len(newK8sVersion) != 0 {
		cfg.KubernetesVersion = newK8sVersion
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

	return client, cfg, NewOfflineVersionGetter(NewKubeVersionGetter(client), newK8sVersion), nil
}

// EnforceRequirements verifies that it's okay to upgrade and then returns the variables needed for the rest of the procedure
func EnforceRequirements(client clientset.Interface, cfg *kubeadmapi.InitConfiguration, ignorePreflightErrorsSet sets.String, featureGatesString string, dryRun bool) error {
	// Check if the cluster is self-hosted
	if IsControlPlaneSelfHosted(client) {
		return errors.New("cannot upgrade a self-hosted control plane")
	}

	// Fetch the configuration from a file or ConfigMap and validate it
	fmt.Println("[upgrade/config] Making sure the configuration is correct:")

	var err error
	// If features gates are passed to the command line, use it (otherwise use featureGates from configuration)
	if featureGatesString != "" {
		cfg.FeatureGates, err = features.NewFeatureGate(&features.InitFeatureGates, featureGatesString)
		if err != nil {
			return errors.Wrap(err, "[upgrade/config] FATAL")
		}
	}

	// Check if feature gate flags used in the cluster are consistent with the set of features currently supported by kubeadm
	if msg := features.CheckDeprecatedFlags(&features.InitFeatureGates, cfg.FeatureGates); len(msg) > 0 {
		for _, m := range msg {
			fmt.Printf("[upgrade/config] %s\n", m)
		}
		return errors.New("[upgrade/config] FATAL. Unable to upgrade a cluster using deprecated feature-gate flags. Please see the release notes")
	}

	// Ensure the user is root
	klog.V(1).Info("running preflight checks")
	if err := RunPreflightChecks(client, ignorePreflightErrorsSet, &cfg.ClusterConfiguration); err != nil {
		return err
	}

	// Run healthchecks against the cluster
	if err := CheckClusterHealth(client, &cfg.ClusterConfiguration, ignorePreflightErrorsSet); err != nil {
		return errors.Wrap(err, "[upgrade/health] FATAL")
	}

	// Use a real version getter interface that queries the API server, the kubeadm client and the Kubernetes CI system for latest versions
	return nil
}

// GetClient gets a real or fake client depending on whether the user is dry-running or not
func GetClient(file string, dryRun bool) (clientset.Interface, error) {
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

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

package apply

import (
	"fmt"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
)

// NewPreflightPhase creates a kubeadm workflow phase that implements preflight checks for upgrade apply
func NewPreflightPhase() workflow.Phase {
	return workflow.Phase{
		Name:  "preflight",
		Short: "Run upgrade apply pre-flight checks",
		Long:  "Run pre-flight checks for kubeadm upgrade apply.",
		Run:   runPreflight,
		InheritFlags: []string{
			options.IgnorePreflightErrors,
		},
	}
}

// EnforceVersionPolicies makes sure that the version the user specified is valid to upgrade to
// There are both fatal and skippable (with --force) errors
func EnforceVersionPolicies(newK8sVersionStr string, newK8sVersion *version.Version, allowExperimentalUpgrades, allowRCUpgrades, force bool, versionGetter upgrade.VersionGetter) error {
	fmt.Printf("[upgrade/version] You have chosen to change the cluster version to %q\n", newK8sVersionStr)

	versionSkewErrs := upgrade.EnforceVersionPolicies(versionGetter, newK8sVersionStr, newK8sVersion, allowExperimentalUpgrades, allowRCUpgrades)
	if versionSkewErrs != nil {

		if len(versionSkewErrs.Mandatory) > 0 {
			return errors.Errorf("the --version argument is invalid due to these fatal errors:\n\n%v\nPlease fix the misalignments highlighted above and try upgrading again",
				kubeadmutil.FormatErrMsg(versionSkewErrs.Mandatory))
		}

		if len(versionSkewErrs.Skippable) > 0 {
			// Return the error if the user hasn't specified the --force flag
			if !force {
				return errors.Errorf("the --version argument is invalid due to these errors:\n\n%v\nCan be bypassed if you pass the --force flag",
					kubeadmutil.FormatErrMsg(versionSkewErrs.Skippable))
			}
			// Soft errors found, but --force was specified
			fmt.Printf("[upgrade/version] Found %d potential version compatibility errors but skipping since the --force flag is set: \n\n%v", len(versionSkewErrs.Skippable), kubeadmutil.FormatErrMsg(versionSkewErrs.Skippable))
		}
	}
	return nil
}

// runPreflight executes preflight checks logic.
func runPreflight(c workflow.RunData) error {
	data, ok := c.(Data)
	if !ok {
		return errors.New("preflight phase invoked with an invalid data struct")
	}
	fmt.Println("[preflight] Running pre-flight checks")

	cfg := data.Cfg()
	client := data.Client()

	// First, check if we're root separately from the other preflight checks and fail fast
	if err := preflight.RunRootCheckOnly(data.IgnorePreflightErrors()); err != nil {
		return err
	}

	if err := upgrade.RunCoreDNSMigrationCheck(client, data.IgnorePreflightErrors()); err != nil {
		return err
	}

	// Run healthchecks against the cluster
	if err := upgrade.CheckClusterHealth(client, &cfg.ClusterConfiguration, data.IgnorePreflightErrors()); err != nil {
		// errors.Wrap(err, "[upgrade/health] FATAL")
		return err
	}

	// Check if feature gate flags used in the cluster are consistent with the set of features currently supported by kubeadm
	if msg := features.CheckDeprecatedFlags(&features.InitFeatureGates, cfg.FeatureGates); len(msg) > 0 {
		for _, m := range msg {
			fmt.Printf("[upgrade/config] %s\n", m)
		}
		return errors.New("[upgrade/config] FATAL. Unable to upgrade a cluster using deprecated feature-gate flags. Please see the release notes")
	}

	// Validate requested and validate actual version
	klog.V(1).Infoln("[upgrade/apply] validating requested and actual version")
	if err := configutil.NormalizeKubernetesVersion(&cfg.ClusterConfiguration); err != nil {
		return err
	}

	// Use normalized version string in all following code.
	newK8sVersion, err := version.ParseSemantic(cfg.KubernetesVersion)
	if err != nil {
		return errors.Errorf("unable to parse normalized version %q as a semantic version", cfg.KubernetesVersion)
	}

	if err := features.ValidateVersion(features.InitFeatureGates, cfg.FeatureGates, cfg.KubernetesVersion); err != nil {
		return err
	}

	// Enforce the version skew policies
	klog.V(1).Infoln("[upgrade/version] enforcing version skew policies")
	if err := EnforceVersionPolicies(cfg.KubernetesVersion, newK8sVersion, false, false, data.Force(), data.VersionGetter()); err != nil {
		return errors.Wrap(err, "[upgrade/version] FATAL")
	}

	// If the current session is interactive, ask the user whether they really want to upgrade.
	if data.sessionIsInteractive() {
		if err := InteractivelyConfirmUpgrade("Are you sure you want to proceed with the upgrade?"); err != nil {
			return err
		}
	}

	if !flags.dryRun {
		fmt.Println("[upgrade/prepull] Pulling images required for setting up a Kubernetes cluster")
		fmt.Println("[upgrade/prepull] This might take a minute or two, depending on the speed of your internet connection")
		fmt.Println("[upgrade/prepull] You can also perform this action in beforehand using 'kubeadm config images pull'")
		if err := preflight.RunPullImagesCheck(utilsexec.New(), cfg, sets.NewString(cfg.NodeRegistration.IgnorePreflightErrors...)); err != nil {
			return err
		}
	} else {
		fmt.Println("[upgrade/prepull] Would pull the required images (like 'kubeadm config images pull')")
	}

	// // if this is a control-plane node, pull the basic images
	// if data.IsControlPlaneNode() {
	// 	if !data.DryRun() {
	// 		fmt.Println("[preflight] Pulling images required for setting up a Kubernetes cluster")
	// 		fmt.Println("[preflight] This might take a minute or two, depending on the speed of your internet connection")
	// 		fmt.Println("[preflight] You can also perform this action in beforehand using 'kubeadm config images pull'")
	// 		if err := preflight.RunPullImagesCheck(utilsexec.New(), data.Cfg(), data.IgnorePreflightErrors()); err != nil {
	// 			return err
	// 		}
	// 	} else {
	// 		fmt.Println("[preflight] Would pull the required images (like 'kubeadm config images pull')")
	// 	}
	// } else {
	// 	fmt.Println("[preflight] Skipping prepull. Not a control plane node.")
	// 	return nil
	// }

	return nil
}

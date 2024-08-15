/*
Copyright 2024 The Kubernetes Authors.

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

// Package apply implements phases of 'kubeadm upgrade apply'.
package apply

import (
	"fmt"
	"os"

	"github.com/pkg/errors"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/klog/v2"
	utilsexec "k8s.io/utils/exec"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

// NewPreflightPhase returns a new prefight phase.
func NewPreflightPhase() workflow.Phase {
	return workflow.Phase{
		Name:  "preflight",
		Short: "Run upgrade apply preflight checks",
		Long:  "Run preflight checks for kubeadm upgrade apply.",
		Run:   runPreflight,
		InheritFlags: []string{
			options.CfgPath,
			options.KubeconfigPath,
			options.DryRun,
			options.IgnorePreflightErrors,
			"allow-experimental-upgrades",
			"allow-release-candidate-upgrades",
			"force",
			"yes",
		},
	}
}

func runPreflight(c workflow.RunData) error {
	data, ok := c.(Data)
	if !ok {
		return errors.New("preflight phase invoked with an invalid data struct")
	}
	fmt.Println("[preflight] Running preflight checks")

	printer := &output.TextPrinter{}

	initCfg, client, ignorePreflightErrors := data.InitCfg(), data.Client(), data.IgnorePreflightErrors()

	// First, check if we're root separately from the other preflight checks and fail fast
	if err := preflight.RunRootCheckOnly(ignorePreflightErrors); err != nil {
		return err
	}

	// Run CoreDNS migration check
	if err := upgrade.RunCoreDNSMigrationCheck(client, ignorePreflightErrors); err != nil {
		return err
	}

	// Run healthchecks against the cluster
	klog.V(1).Infoln("[upgrade/apply] Verifying the cluster health")
	if err := upgrade.CheckClusterHealth(client, &initCfg.ClusterConfiguration, ignorePreflightErrors, printer); err != nil {
		return err
	}

	// Check if feature gate flags used in the cluster are consistent with the set of features currently supported by kubeadm
	if msg := features.CheckDeprecatedFlags(&features.InitFeatureGates, initCfg.FeatureGates); len(msg) > 0 {
		for _, m := range msg {
			_, _ = printer.Printf("[upgrade/config] %s\n", m)
		}
	}

	// Validate requested and validate actual version
	klog.V(1).Infoln("[upgrade/apply] Validating requested and actual version")
	if err := configutil.NormalizeKubernetesVersion(&initCfg.ClusterConfiguration); err != nil {
		return err
	}

	// Use normalized version string in all following code.
	upgradeVersion, err := version.ParseSemantic(initCfg.KubernetesVersion)
	if err != nil {
		return errors.Errorf("unable to parse normalized version %q as a semantic version", initCfg.KubernetesVersion)
	}

	if err := features.ValidateVersion(features.InitFeatureGates, initCfg.FeatureGates, initCfg.KubernetesVersion); err != nil {
		return err
	}

	versionGetter := upgrade.NewOfflineVersionGetter(upgrade.NewKubeVersionGetter(client), initCfg.KubernetesVersion)
	if err := enforceVersionPolicies(initCfg.KubernetesVersion, upgradeVersion, data.AllowExperimentalUpgrades(), data.AllowRCUpgrades(), data.ForceUpgrade(), versionGetter); err != nil {
		return err
	}

	if data.SessionIsInteractive() {
		if err := cmdutil.InteractivelyConfirmAction("upgrade", "Are you sure you want to proceed?", os.Stdin); err != nil {
			return err
		}
	}

	if !data.DryRun() {
		fmt.Println("[preflight] Pulling images required for setting up a Kubernetes cluster")
		fmt.Println("[preflight] This might take a minute or two, depending on the speed of your internet connection")
		fmt.Println("[preflight] You can also perform this action beforehand using 'kubeadm config images pull'")
		if err := preflight.RunPullImagesCheck(utilsexec.New(), initCfg, ignorePreflightErrors); err != nil {
			return err
		}
	} else {
		fmt.Println("[preflight] Would pull the required images (like 'kubeadm config images pull')")
	}

	return nil
}

// enforceVersionPolicies makes sure that the version the user specified is valid to upgrade to
// There are both fatal and skippable (with --force) errors
func enforceVersionPolicies(newK8sVersionStr string, newK8sVersion *version.Version, allowExperimentalUpgrades, allowRCUpgrades, force bool, versionGetter upgrade.VersionGetter) error {
	fmt.Printf("[upgrade/version] You have chosen to upgrade the cluster version to %q\n", newK8sVersionStr)

	versionSkewErrs := upgrade.EnforceVersionPolicies(versionGetter, newK8sVersionStr, newK8sVersion, allowExperimentalUpgrades, allowRCUpgrades)
	if versionSkewErrs != nil {

		if len(versionSkewErrs.Mandatory) > 0 {
			return errors.Errorf("the version argument is invalid due to these fatal errors:\n\n%v\nPlease fix the misalignments highlighted above and try upgrading again",
				kubeadmutil.FormatErrMsg(versionSkewErrs.Mandatory))
		}

		if len(versionSkewErrs.Skippable) > 0 {
			// Return the error if the user hasn't specified the --force flag
			if !force {
				return errors.Errorf("the version argument is invalid due to these errors:\n\n%v\nCan be bypassed if you pass the --force flag",
					kubeadmutil.FormatErrMsg(versionSkewErrs.Skippable))
			}
			// Soft errors found, but --force was specified
			fmt.Printf("[upgrade/version] Found %d potential version compatibility errors but skipping since the --force flag is set: \n\n%v", len(versionSkewErrs.Skippable), kubeadmutil.FormatErrMsg(versionSkewErrs.Skippable))
		}
	}
	return nil
}

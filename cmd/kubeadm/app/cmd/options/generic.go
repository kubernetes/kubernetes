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

package options

import (
	"strings"

	"github.com/spf13/pflag"

	cliflag "k8s.io/component-base/cli/flag"

	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/features"
)

// AddKubeConfigFlag adds the --kubeconfig flag to the given flagset
func AddKubeConfigFlag(fs *pflag.FlagSet, kubeConfigFile *string) {
	fs.StringVar(kubeConfigFile, KubeconfigPath, *kubeConfigFile, "The kubeconfig file to use when talking to the cluster. If the flag is not set, a set of standard locations can be searched for an existing kubeconfig file.")
	// Note that DefValue is the text shown in the terminal and not the default value assigned to the flag
	fs.Lookup(KubeconfigPath).DefValue = constants.GetAdminKubeConfigPath()
}

// AddKubeConfigDirFlag adds the --kubeconfig-dir flag to the given flagset
func AddKubeConfigDirFlag(fs *pflag.FlagSet, kubeConfigDir *string) {
	fs.StringVar(kubeConfigDir, KubeconfigDir, *kubeConfigDir, "The path where to save the kubeconfig file.")
}

// AddConfigFlag adds the --config flag to the given flagset
func AddConfigFlag(fs *pflag.FlagSet, cfgPath *string) {
	fs.StringVar(cfgPath, CfgPath, *cfgPath, "Path to a kubeadm configuration file.")
}

// AddIgnorePreflightErrorsFlag adds the --ignore-preflight-errors flag to the given flagset
func AddIgnorePreflightErrorsFlag(fs *pflag.FlagSet, ignorePreflightErrors *[]string) {
	fs.StringSliceVar(
		ignorePreflightErrors, IgnorePreflightErrors, *ignorePreflightErrors,
		"A list of checks whose errors will be shown as warnings. Example: 'IsPrivilegedUser,Swap'. Value 'all' ignores errors from all checks.",
	)
}

// AddControlPlanExtraArgsFlags adds the ExtraArgs flags for control plane components
func AddControlPlanExtraArgsFlags(fs *pflag.FlagSet, apiServerExtraArgs, controllerManagerExtraArgs, schedulerExtraArgs *map[string]string) {
	// TODO: https://github.com/kubernetes/kubeadm/issues/1601
	// Either deprecate these flags or handle duplicate keys.
	// Currently the map[string]string returned by NewMapStringString() doesn't allow this.
	fs.Var(cliflag.NewMapStringString(apiServerExtraArgs), APIServerExtraArgs, "A set of extra flags to pass to the API Server or override default ones in form of <flagname>=<value>")
	fs.Var(cliflag.NewMapStringString(controllerManagerExtraArgs), ControllerManagerExtraArgs, "A set of extra flags to pass to the Controller Manager or override default ones in form of <flagname>=<value>")
	fs.Var(cliflag.NewMapStringString(schedulerExtraArgs), SchedulerExtraArgs, "A set of extra flags to pass to the Scheduler or override default ones in form of <flagname>=<value>")
}

// AddImageMetaFlags adds the --image-repository flag to the given flagset
func AddImageMetaFlags(fs *pflag.FlagSet, imageRepository *string) {
	fs.StringVar(imageRepository, ImageRepository, *imageRepository, "Choose a container registry to pull control plane images from")
}

// AddFeatureGatesStringFlag adds the --feature-gates flag to the given flagset
func AddFeatureGatesStringFlag(fs *pflag.FlagSet, featureGatesString *string) {
	if knownFeatures := features.KnownFeatures(&features.InitFeatureGates); len(knownFeatures) > 0 {
		fs.StringVar(featureGatesString, FeatureGatesString, *featureGatesString, "A set of key=value pairs that describe feature gates for various features. "+
			"Options are:\n"+strings.Join(knownFeatures, "\n"))
	} else {
		fs.StringVar(featureGatesString, FeatureGatesString, *featureGatesString, "A set of key=value pairs that describe feature gates for various features. "+
			"No feature gates are available in this release.")
	}
}

// AddKubernetesVersionFlag adds the --kubernetes-version flag to the given flagset
func AddKubernetesVersionFlag(fs *pflag.FlagSet, kubernetesVersion *string) {
	fs.StringVar(
		kubernetesVersion, KubernetesVersion, *kubernetesVersion,
		`Choose a specific Kubernetes version for the control plane.`,
	)
}

// AddKubeadmOtherFlags adds flags that are not bound to a configuration file to the given flagset
func AddKubeadmOtherFlags(flagSet *pflag.FlagSet, rootfsPath *string) {
	flagSet.StringVar(
		rootfsPath, "rootfs", *rootfsPath,
		"The path to the 'real' host root filesystem. This will cause kubeadm to chroot into the provided path.",
	)
}

// AddPatchesFlag adds the --patches flag to the given flagset
func AddPatchesFlag(fs *pflag.FlagSet, patchesDir *string) {
	const usage = `Path to a directory that contains files named ` +
		`"target[suffix][+patchtype].extension". For example, ` +
		`"kube-apiserver0+merge.yaml" or just "etcd.json". ` +
		`"target" can be one of "kube-apiserver", "kube-controller-manager", "kube-scheduler", "etcd", "kubeletconfiguration". ` +
		`"patchtype" can be one of "strategic", "merge" or "json" and they match the patch formats ` +
		`supported by kubectl. The default "patchtype" is "strategic". "extension" must be either ` +
		`"json" or "yaml". "suffix" is an optional string that can be used to determine ` +
		`which patches are applied first alpha-numerically.`
	fs.StringVar(patchesDir, Patches, *patchesDir, usage)
}

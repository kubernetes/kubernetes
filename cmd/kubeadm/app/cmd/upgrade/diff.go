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

package upgrade

import (
	"io"
	"os"

	"github.com/pkg/errors"
	"github.com/pmezard/go-difflib/difflib"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	kubeconfigutil "k8s.io/kubernetes/cmd/kubeadm/app/util/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/output"
)

type diffFlags struct {
	apiServerManifestPath         string
	controllerManagerManifestPath string
	schedulerManifestPath         string
	newK8sVersionStr              string
	contextLines                  int
	kubeConfigPath                string
	cfgPath                       string
	out                           io.Writer
}

var (
	defaultAPIServerManifestPath         = constants.GetStaticPodFilepath(constants.KubeAPIServer, constants.GetStaticPodDirectory())
	defaultControllerManagerManifestPath = constants.GetStaticPodFilepath(constants.KubeControllerManager, constants.GetStaticPodDirectory())
	defaultSchedulerManifestPath         = constants.GetStaticPodFilepath(constants.KubeScheduler, constants.GetStaticPodDirectory())
)

// newCmdDiff returns the cobra command for `kubeadm upgrade diff`
func newCmdDiff(out io.Writer) *cobra.Command {
	flags := &diffFlags{
		kubeConfigPath: constants.GetAdminKubeConfigPath(),
		out:            out,
	}

	cmd := &cobra.Command{
		Use:   "diff [version]",
		Short: "Show what differences would be applied to existing static pod manifests. See also: kubeadm upgrade apply --dry-run",
		RunE: func(cmd *cobra.Command, args []string) error {
			// Run preflight checks for diff to check that the manifests already exist.
			if err := validateManifestsPath(
				flags.apiServerManifestPath,
				flags.controllerManagerManifestPath,
				flags.schedulerManifestPath); err != nil {
				return err
			}

			if err := validation.ValidateMixedArguments(cmd.Flags()); err != nil {
				return err
			}
			return runDiff(cmd.Flags(), flags, args, configutil.FetchInitConfigurationFromCluster)
		},
	}

	options.AddKubeConfigFlag(cmd.Flags(), &flags.kubeConfigPath)
	options.AddConfigFlag(cmd.Flags(), &flags.cfgPath)
	cmd.Flags().StringVar(&flags.apiServerManifestPath, "api-server-manifest", defaultAPIServerManifestPath, "path to API server manifest")
	cmd.Flags().StringVar(&flags.controllerManagerManifestPath, "controller-manager-manifest", defaultControllerManagerManifestPath, "path to controller manifest")
	cmd.Flags().StringVar(&flags.schedulerManifestPath, "scheduler-manifest", defaultSchedulerManifestPath, "path to scheduler manifest")
	cmd.Flags().IntVarP(&flags.contextLines, "context-lines", "c", 3, "How many lines of context in the diff")

	return cmd
}

func validateManifestsPath(manifests ...string) (err error) {
	for _, manifestPath := range manifests {
		if len(manifestPath) == 0 {
			return errors.New("empty manifest path")
		}
		s, err := os.Stat(manifestPath)
		if err != nil {
			if os.IsNotExist(err) {
				return errors.Wrapf(err, "the manifest file %q does not exist", manifestPath)
			}
			return errors.Wrapf(err, "error obtaining stats for manifest file %q", manifestPath)
		}
		if s.IsDir() {
			return errors.Errorf("%q is a directory", manifestPath)
		}
	}
	return nil
}

// FetchInitConfigurationFunc defines the signature of the function which will fetch InitConfiguration from cluster.
type FetchInitConfigurationFunc func(client clientset.Interface, printer output.Printer, logPrefix string, newControlPlane, skipComponentConfigs bool) (*kubeadmapi.InitConfiguration, error)

func runDiff(fs *pflag.FlagSet, flags *diffFlags, args []string, fetchInitConfigurationFromCluster FetchInitConfigurationFunc) error {
	externalCfg := &v1beta4.UpgradeConfiguration{}
	opt := configutil.LoadOrDefaultConfigurationOptions{}
	upgradeCfg, err := configutil.LoadOrDefaultUpgradeConfiguration(flags.cfgPath, externalCfg, opt)
	if err != nil {
		return err
	}
	client, err := kubeconfigutil.ClientSetFromFile(flags.kubeConfigPath)
	if err != nil {
		return errors.Wrapf(err, "couldn't create a Kubernetes client from file %q", flags.kubeConfigPath)
	}
	initCfg, err := fetchInitConfigurationFromCluster(client, &output.TextPrinter{}, "upgrade/diff", false, true)
	if err != nil {
		return err
	}

	// Pick up the version from the ClusterConfiguration.
	if initCfg.KubernetesVersion != "" {
		flags.newK8sVersionStr = initCfg.KubernetesVersion
	}
	if upgradeCfg.Diff.KubernetesVersion != "" {
		flags.newK8sVersionStr = upgradeCfg.Diff.KubernetesVersion
	}

	// Version must be specified via version arg if it's not set in ClusterConfiguration.
	if flags.newK8sVersionStr == "" {
		if err := cmdutil.ValidateExactArgNumber(args, []string{"version"}); err != nil {
			return err
		}
	}
	// If option was specified in both args and config file, args will overwrite the config file.
	if len(args) == 1 {
		flags.newK8sVersionStr = args[0]
	}
	_, err = version.ParseSemantic(flags.newK8sVersionStr)
	if err != nil {
		return err
	}

	initCfg.ClusterConfiguration.KubernetesVersion = flags.newK8sVersionStr

	specs := controlplane.GetStaticPodSpecs(&initCfg.ClusterConfiguration, &initCfg.LocalAPIEndpoint, nil)
	for spec, pod := range specs {
		var path string
		switch spec {
		case constants.KubeAPIServer:
			path = flags.apiServerManifestPath
		case constants.KubeControllerManager:
			path = flags.controllerManagerManifestPath
		case constants.KubeScheduler:
			path = flags.schedulerManifestPath
		default:
			klog.Errorf("[diff] unknown spec %v", spec)
			continue
		}
		newManifest, err := kubeadmutil.MarshalToYaml(&pod, corev1.SchemeGroupVersion)
		if err != nil {
			return err
		}
		if path == "" {
			return errors.New("empty manifest path")
		}
		existingManifest, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		// Populated and write out the diff
		diff := difflib.UnifiedDiff{
			A:        difflib.SplitLines(string(existingManifest)),
			B:        difflib.SplitLines(string(newManifest)),
			FromFile: path,
			ToFile:   "new manifest",
			Context:  cmdutil.ValueFromFlagsOrConfig(fs, "context-lines", upgradeCfg.Diff.DiffContextLines, flags.contextLines).(int),
		}

		difflib.WriteUnifiedDiff(flags.out, diff)
	}
	return nil
}

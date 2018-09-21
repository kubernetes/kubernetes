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
	"fmt"
	"io"
	"io/ioutil"

	"github.com/golang/glog"
	"github.com/pmezard/go-difflib/difflib"
	"github.com/spf13/cobra"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/version"
	kubeadmapiv1alpha3 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha3"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

type diffFlags struct {
	apiServerManifestPath         string
	controllerManagerManifestPath string
	schedulerManifestPath         string
	newK8sVersionStr              string
	contextLines                  int
	cfgPath                       string
	out                           io.Writer
}

var (
	defaultAPIServerManifestPath         = constants.GetStaticPodFilepath(constants.KubeAPIServer, constants.GetStaticPodDirectory())
	defaultControllerManagerManifestPath = constants.GetStaticPodFilepath(constants.KubeControllerManager, constants.GetStaticPodDirectory())
	defaultSchedulerManifestPath         = constants.GetStaticPodFilepath(constants.KubeScheduler, constants.GetStaticPodDirectory())
)

// NewCmdDiff returns the cobra command for `kubeadm upgrade diff`
func NewCmdDiff(out io.Writer) *cobra.Command {
	flags := &diffFlags{
		out: out,
	}

	cmd := &cobra.Command{
		Use:   "diff [version]",
		Short: "Show what differences would be applied to existing static pod manifests. See also: kubeadm upgrade apply --dry-run",
		Run: func(cmd *cobra.Command, args []string) {
			// TODO: Run preflight checks for diff to check that the manifests already exist.
			kubeadmutil.CheckErr(runDiff(flags, args))
		},
	}

	options.AddConfigFlag(cmd.Flags(), &flags.cfgPath)
	cmd.Flags().StringVar(&flags.apiServerManifestPath, "api-server-manifest", defaultAPIServerManifestPath, "path to API server manifest")
	cmd.Flags().StringVar(&flags.controllerManagerManifestPath, "controller-manager-manifest", defaultControllerManagerManifestPath, "path to controller manifest")
	cmd.Flags().StringVar(&flags.schedulerManifestPath, "scheduler-manifest", defaultSchedulerManifestPath, "path to scheduler manifest")
	cmd.Flags().IntVarP(&flags.contextLines, "context-lines", "c", 3, "How many lines of context in the diff")

	return cmd
}

func runDiff(flags *diffFlags, args []string) error {

	// If the version is specified in config file, pick up that value.
	glog.V(1).Infof("fetching configuration from file %s", flags.cfgPath)
	cfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(flags.cfgPath, &kubeadmapiv1alpha3.InitConfiguration{})
	if err != nil {
		return err
	}

	if cfg.KubernetesVersion != "" {
		flags.newK8sVersionStr = cfg.KubernetesVersion
	}

	// If the new version is already specified in config file, version arg is optional.
	if flags.newK8sVersionStr == "" {
		if err := cmdutil.ValidateExactArgNumber(args, []string{"version"}); err != nil {
			return err
		}
	}

	// If option was specified in both args and config file, args will overwrite the config file.
	if len(args) == 1 {
		flags.newK8sVersionStr = args[0]
	}

	k8sVer, err := version.ParseSemantic(flags.newK8sVersionStr)
	if err != nil {
		return err
	}

	specs := controlplane.GetStaticPodSpecs(cfg, k8sVer)
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
			glog.Errorf("[diff] unknown spec %v", spec)
			continue
		}

		newManifest, err := kubeadmutil.MarshalToYaml(&pod, corev1.SchemeGroupVersion)
		if err != nil {
			return err
		}
		if path == "" {
			return fmt.Errorf("empty manifest path")
		}
		existingManifest, err := ioutil.ReadFile(path)
		if err != nil {
			return err
		}

		// Populated and write out the diff
		diff := difflib.UnifiedDiff{
			A:        difflib.SplitLines(string(existingManifest)),
			B:        difflib.SplitLines(string(newManifest)),
			FromFile: path,
			ToFile:   "new manifest",
			Context:  flags.contextLines,
		}

		difflib.WriteUnifiedDiff(flags.out, diff)
	}
	return nil
}

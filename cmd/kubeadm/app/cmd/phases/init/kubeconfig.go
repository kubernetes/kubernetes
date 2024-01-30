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

package phases

import (
	"fmt"
	"path/filepath"

	"github.com/pkg/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

var (
	kubeconfigFilePhaseProperties = map[string]struct {
		name  string
		short string
		long  string
	}{
		kubeadmconstants.AdminKubeConfigFileName: {
			name:  "admin",
			short: "Generate a kubeconfig file for the admin to use and for kubeadm itself",
			long:  "Generate the kubeconfig file for the admin and for kubeadm itself, and save it to %s file.",
		},
		kubeadmconstants.SuperAdminKubeConfigFileName: {
			name:  "super-admin",
			short: "Generate a kubeconfig file for the super-admin",
			long:  "Generate a kubeconfig file for the super-admin, and save it to %s file.",
		},
		kubeadmconstants.KubeletKubeConfigFileName: {
			name:  "kubelet",
			short: "Generate a kubeconfig file for the kubelet to use *only* for cluster bootstrapping purposes",
			long: cmdutil.LongDesc(`
					Generate the kubeconfig file for the kubelet to use and save it to %s file.

					Please note that this should *only* be used for cluster bootstrapping purposes. After your control plane is up,
					you should request all kubelet credentials from the CSR API.`),
		},
		kubeadmconstants.ControllerManagerKubeConfigFileName: {
			name:  "controller-manager",
			short: "Generate a kubeconfig file for the controller manager to use",
			long:  "Generate the kubeconfig file for the controller manager to use and save it to %s file",
		},
		kubeadmconstants.SchedulerKubeConfigFileName: {
			name:  "scheduler",
			short: "Generate a kubeconfig file for the scheduler to use",
			long:  "Generate the kubeconfig file for the scheduler to use and save it to %s file.",
		},
	}
)

// NewKubeConfigPhase creates a kubeadm workflow phase that creates all kubeconfig files necessary to establish the control plane and the admin kubeconfig file.
func NewKubeConfigPhase() workflow.Phase {
	return workflow.Phase{
		Name:  "kubeconfig",
		Short: "Generate all kubeconfig files necessary to establish the control plane and the admin kubeconfig file",
		Long:  cmdutil.MacroCommandLongDescription,
		Phases: []workflow.Phase{
			{
				Name:           "all",
				Short:          "Generate all kubeconfig files",
				InheritFlags:   getKubeConfigPhaseFlags("all"),
				RunAllSiblings: true,
			},
			NewKubeConfigFilePhase(kubeadmconstants.AdminKubeConfigFileName),
			NewKubeConfigFilePhase(kubeadmconstants.SuperAdminKubeConfigFileName),
			NewKubeConfigFilePhase(kubeadmconstants.KubeletKubeConfigFileName),
			NewKubeConfigFilePhase(kubeadmconstants.ControllerManagerKubeConfigFileName),
			NewKubeConfigFilePhase(kubeadmconstants.SchedulerKubeConfigFileName),
		},
		Run: runKubeConfig,
	}
}

// NewKubeConfigFilePhase creates a kubeadm workflow phase that creates a kubeconfig file.
func NewKubeConfigFilePhase(kubeConfigFileName string) workflow.Phase {
	return workflow.Phase{
		Name:         kubeconfigFilePhaseProperties[kubeConfigFileName].name,
		Short:        kubeconfigFilePhaseProperties[kubeConfigFileName].short,
		Long:         fmt.Sprintf(kubeconfigFilePhaseProperties[kubeConfigFileName].long, kubeConfigFileName),
		Run:          runKubeConfigFile(kubeConfigFileName),
		InheritFlags: getKubeConfigPhaseFlags(kubeConfigFileName),
	}
}

func getKubeConfigPhaseFlags(name string) []string {
	flags := []string{
		options.APIServerAdvertiseAddress,
		options.ControlPlaneEndpoint,
		options.APIServerBindPort,
		options.CertificatesDir,
		options.CfgPath,
		options.KubeconfigDir,
		options.KubernetesVersion,
		options.DryRun,
	}
	if name == "all" || name == kubeadmconstants.KubeletKubeConfigFileName {
		flags = append(flags,
			options.NodeName,
		)
	}
	return flags
}

func runKubeConfig(c workflow.RunData) error {
	data, ok := c.(InitData)
	if !ok {
		return errors.New("kubeconfig phase invoked with an invalid data struct")
	}

	fmt.Printf("[kubeconfig] Using kubeconfig folder %q\n", data.KubeConfigDir())
	return nil
}

// runKubeConfigFile executes kubeconfig creation logic.
func runKubeConfigFile(kubeConfigFileName string) func(workflow.RunData) error {
	return func(c workflow.RunData) error {
		data, ok := c.(InitData)
		if !ok {
			return errors.New("kubeconfig phase invoked with an invalid data struct")
		}

		// if external CA mode, skip certificate authority generation
		if data.ExternalCA() {
			fmt.Printf("[kubeconfig] External CA mode: Using user provided %s\n", kubeConfigFileName)
			// If using an external CA while dryrun, copy kubeconfig files to dryrun dir for later use
			if data.DryRun() {
				err := kubeadmutil.CopyFile(filepath.Join(kubeadmconstants.KubernetesDir, kubeConfigFileName), filepath.Join(data.KubeConfigDir(), kubeConfigFileName))
				if err != nil {
					return errors.Wrapf(err, "could not copy %s to dry run directory %s", kubeConfigFileName, data.KubeConfigDir())
				}
			}
			return nil
		}

		// if dryrunning, reads certificates from a temporary folder (and defer restore to the path originally specified by the user)
		cfg := data.Cfg()
		cfg.CertificatesDir = data.CertificateWriteDir()
		defer func() { cfg.CertificatesDir = data.CertificateDir() }()

		// creates the KubeConfig file (or use existing)
		return kubeconfigphase.CreateKubeConfigFile(kubeConfigFileName, data.KubeConfigDir(), data.Cfg())
	}
}

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

	"github.com/pkg/errors"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	kubeconfigFilePhaseProperties = map[string]struct {
		name  string
		short string
		long  string
	}{
		kubeadmconstants.AdminKubeConfigFileName: {
			name:  "admin",
			short: "Generates a kubeconfig file for the admin to use and for kubeadm itself",
			long:  "Generates the kubeconfig file for the admin and for kubeadm itself, and saves it to %s file.",
		},
		kubeadmconstants.KubeletKubeConfigFileName: {
			name:  "kubelet",
			short: "Generates a kubeconfig file for the kubelet to use *only* for cluster bootstrapping purposes",
			long: normalizer.LongDesc(`
					Generates the kubeconfig file for the kubelet to use and saves it to %s file.
			
					Please note that this should *only* be used for cluster bootstrapping purposes. After your control plane is up,
					you should request all kubelet credentials from the CSR API.`),
		},
		kubeadmconstants.ControllerManagerKubeConfigFileName: {
			name:  "controller-manager",
			short: "Generates a kubeconfig file for the controller manager to use",
			long:  "Generates the kubeconfig file for the controller manager to use and saves it to %s file",
		},
		kubeadmconstants.SchedulerKubeConfigFileName: {
			name:  "scheduler",
			short: "Generates a kubeconfig file for the scheduler to use",
			long:  "Generates the kubeconfig file for the scheduler to use and saves it to %s file.",
		},
	}
)

// kubeConfigData defines the behavior that a runtime data struct passed to the kubeconfig phase
// should have. Please note that we are using an interface in order to make this phase reusable in different workflows
// (and thus with different runtime data struct, all of them requested to be compliant to this interface)
type kubeConfigData interface {
	Cfg() *kubeadmapi.InitConfiguration
	ExternalCA() bool
	CertificateDir() string
	CertificateWriteDir() string
	KubeConfigDir() string
}

// NewKubeConfigPhase creates a kubeadm workflow phase that creates all kubeconfig files necessary to establish the control plane and the admin kubeconfig file.
func NewKubeConfigPhase() workflow.Phase {
	return workflow.Phase{
		Name:  "kubeconfig",
		Short: "Generates all kubeconfig files necessary to establish the control plane and the admin kubeconfig file",
		Long:  cmdutil.MacroCommandLongDescription,
		Phases: []workflow.Phase{
			{
				Name:           "all",
				Short:          "Generates all kubeconfig files",
				InheritFlags:   getKubeConfigPhaseFlags("all"),
				RunAllSiblings: true,
			},
			NewKubeConfigFilePhase(kubeadmconstants.AdminKubeConfigFileName),
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
		options.APIServerBindPort,
		options.CertificatesDir,
		options.CfgPath,
		options.KubeconfigDir,
	}
	if name == "all" || name == kubeadmconstants.KubeletKubeConfigFileName {
		flags = append(flags,
			options.NodeName,
		)
	}
	return flags
}

func runKubeConfig(c workflow.RunData) error {
	data, ok := c.(kubeConfigData)
	if !ok {
		return errors.New("kubeconfig phase invoked with an invalid data struct")
	}

	fmt.Printf("[kubeconfig] Using kubeconfig folder %q\n", data.KubeConfigDir())
	return nil
}

// runKubeConfigFile executes kubeconfig creation logic.
func runKubeConfigFile(kubeConfigFileName string) func(workflow.RunData) error {
	return func(c workflow.RunData) error {
		data, ok := c.(kubeConfigData)
		if !ok {
			return errors.New("kubeconfig phase invoked with an invalid data struct")
		}

		// if external CA mode, skip certificate authority generation
		if data.ExternalCA() {
			fmt.Printf("[kubeconfig] External CA mode: Using user provided %s\n", kubeConfigFileName)
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

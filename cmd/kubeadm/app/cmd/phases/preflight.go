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

package phases

import (
	"fmt"

	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	"k8s.io/kubernetes/pkg/util/normalizer"
	utilsexec "k8s.io/utils/exec"
)

var (
	masterPreflightExample = normalizer.Examples(`
		# Run master pre-flight checks using a config file.
		kubeadm init phase preflight --config kubeadm-config.yml
		`)
)

// preflightMasterData defines the behavior that a runtime data struct passed to the PreflightMaster master phase
// should have. Please note that we are using an interface in order to make this phase reusable in different workflows
// (and thus with different runtime data struct, all of them requested to be compliant to this interface)
type preflightMasterData interface {
	Cfg() *kubeadmapi.InitConfiguration
	DryRun() bool
	IgnorePreflightErrors() sets.String
}

// NewPreflightMasterPhase creates a kubeadm workflow phase that implements preflight checks for a new master node.
func NewPreflightMasterPhase() workflow.Phase {
	return workflow.Phase{
		Name:    "preflight",
		Short:   "Run master pre-flight checks",
		Long:    "Run master pre-flight checks, functionally equivalent to what implemented by kubeadm init.",
		Example: masterPreflightExample,
		Run:     runPreflightMaster,
		InheritFlags: []string{
			options.CfgPath,
			options.IgnorePreflightErrors,
		},
	}
}

// runPreflightMaster executes preflight checks logic.
func runPreflightMaster(c workflow.RunData) error {
	data, ok := c.(preflightMasterData)
	if !ok {
		return errors.New("preflight phase invoked with an invalid data struct")
	}

	fmt.Println("[preflight] Running pre-flight checks")
	if err := preflight.RunInitMasterChecks(utilsexec.New(), data.Cfg(), data.IgnorePreflightErrors()); err != nil {
		return err
	}

	if !data.DryRun() {
		fmt.Println("[preflight] Pulling images required for setting up a Kubernetes cluster")
		fmt.Println("[preflight] This might take a minute or two, depending on the speed of your internet connection")
		fmt.Println("[preflight] You can also perform this action in beforehand using 'kubeadm config images pull'")
		if err := preflight.RunPullImagesCheck(utilsexec.New(), data.Cfg(), data.IgnorePreflightErrors()); err != nil {
			return err
		}
	} else {
		fmt.Println("[preflight] Would pull the required images (like 'kubeadm config images pull')")
	}

	return nil
}

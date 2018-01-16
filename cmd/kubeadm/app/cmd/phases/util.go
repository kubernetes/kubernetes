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
	"github.com/spf13/cobra"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/validation"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
)

// runCmdPhase creates a cobra.Command Run function, by composing the call to the given cmdFunc with necessary additional steps (e.g preparation of input parameters)
func runCmdPhase(cmdFunc func(outDir string, cfg *kubeadmapi.MasterConfiguration) error, outDir, cfgPath *string, cfg *kubeadmapiext.MasterConfiguration) func(cmd *cobra.Command, args []string) {

	// the following statement build a closure that wraps a call to a cmdFunc, binding
	// the function itself with the specific parameters of each sub command.
	// Please note that specific parameter should be passed as value, while other parameters - passed as reference -
	// are shared between sub commands and gets access to current value e.g. flags value.

	return func(cmd *cobra.Command, args []string) {
		if err := validation.ValidateMixedArguments(cmd.Flags()); err != nil {
			kubeadmutil.CheckErr(err)
		}

		// This call returns the ready-to-use configuration based on the configuration file that might or might not exist and the default cfg populated by flags
		internalcfg, err := configutil.ConfigFileAndDefaultsToInternalConfig(*cfgPath, cfg)
		kubeadmutil.CheckErr(err)

		// Execute the cmdFunc
		err = cmdFunc(*outDir, internalcfg)
		kubeadmutil.CheckErr(err)
	}
}

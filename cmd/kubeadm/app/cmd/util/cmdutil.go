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

package util

import (
	"bufio"
	"fmt"
	"io"
	"strings"

	"github.com/pkg/errors"
	"github.com/spf13/cobra"
	"github.com/spf13/pflag"

	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/klog/v2"

	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta3"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

// SubCmdRunE returns a function that handles a case where a subcommand must be specified
// Without this callback, if a user runs just the command without a subcommand,
// or with an invalid subcommand, cobra will print usage information, but still exit cleanly.
// We want to return an error code in these cases so that the
// user knows that their command was invalid.
func SubCmdRunE(name string) func(*cobra.Command, []string) error {
	return func(_ *cobra.Command, args []string) error {
		if len(args) < 1 {
			return errors.Errorf("missing subcommand; %q is not meant to be run on its own", name)
		}

		return errors.Errorf("invalid subcommand: %q", args[0])
	}
}

// ValidateExactArgNumber validates that the required top-level arguments are specified
func ValidateExactArgNumber(args []string, supportedArgs []string) error {
	lenSupported := len(supportedArgs)
	validArgs := 0
	// Disregard possible "" arguments; they are invalid
	for _, arg := range args {
		if len(arg) > 0 {
			validArgs++
		}
		// break early for too many arguments
		if validArgs > lenSupported {
			return errors.Errorf("too many arguments. Required arguments: %v", supportedArgs)
		}
	}

	if validArgs < lenSupported {
		return errors.Errorf("missing one or more required arguments. Required arguments: %v", supportedArgs)
	}
	return nil
}

// GetKubeConfigPath can be used to search for a kubeconfig in standard locations
// if and empty string is passed to the function. If a non-empty string is passed
// the function returns the same string.
func GetKubeConfigPath(file string) string {
	// If a value is provided respect that.
	if file != "" {
		return file
	}
	// Find a config in the standard locations using DefaultClientConfigLoadingRules,
	// but also consider the default config path.
	rules := clientcmd.NewDefaultClientConfigLoadingRules()
	rules.Precedence = append(rules.Precedence, kubeadmconstants.GetAdminKubeConfigPath())
	file = rules.GetDefaultFilename()
	klog.V(1).Infof("Using kubeconfig file: %s", file)
	return file
}

// AddCRISocketFlag adds the cri-socket flag to the supplied flagSet
func AddCRISocketFlag(flagSet *pflag.FlagSet, criSocket *string) {
	flagSet.StringVar(
		criSocket, options.NodeCRISocket, *criSocket,
		"Path to the CRI socket to connect. If empty kubeadm will try to auto-detect this value; use this option only if you have more than one CRI installed or if you have non-standard CRI socket.",
	)
}

// DefaultInitConfiguration return default InitConfiguration. Avoid running the CRI auto-detection
// code as we don't need it.
func DefaultInitConfiguration() *kubeadmapiv1.InitConfiguration {
	initCfg := &kubeadmapiv1.InitConfiguration{
		NodeRegistration: kubeadmapiv1.NodeRegistrationOptions{
			CRISocket: kubeadmconstants.UnknownCRISocket, // avoid CRI detection
		},
	}
	return initCfg
}

// InteractivelyConfirmAction asks the user whether they _really_ want to take the action.
func InteractivelyConfirmAction(action, question string, r io.Reader) error {
	fmt.Printf("[%s] %s [y/N]: ", action, question)
	scanner := bufio.NewScanner(r)
	scanner.Scan()
	if err := scanner.Err(); err != nil {
		return errors.Wrap(err, "couldn't read from standard input")
	}
	answer := scanner.Text()
	if strings.ToLower(answer) == "y" || strings.ToLower(answer) == "yes" {
		return nil
	}

	return errors.New("won't proceed; the user didn't answer (Y|y) in order to continue")
}

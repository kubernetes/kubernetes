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
	"encoding/json"
	"fmt"

	"github.com/spf13/cobra"

	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	clientset "k8s.io/client-go/kubernetes"
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
			return fmt.Errorf("missing subcommand; %q is not meant to be run on its own", name)
		}

		return fmt.Errorf("invalid subcommand: %q", args[0])
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
			return fmt.Errorf("too many arguments. Required arguments: %v", supportedArgs)
		}
	}

	if validArgs < lenSupported {
		return fmt.Errorf("missing one or more required arguments. Required arguments: %v", supportedArgs)
	}
	return nil
}

// UpdateNodeWithConfigMap updates node ConfigSource with KubeletBaseConfigurationConfigMap
func UpdateNodeWithConfigMap(client clientset.Interface, nodeName string) error {
	node, err := client.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	if err != nil {
		return err
	}

	oldData, err := json.Marshal(node)
	if err != nil {
		return err
	}

	kubeletCfg, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(kubeadmconstants.KubeletBaseConfigurationConfigMap, metav1.GetOptions{})
	if err != nil {
		return err
	}

	node.Spec.ConfigSource.ConfigMapRef.UID = kubeletCfg.UID

	newData, err := json.Marshal(node)
	if err != nil {
		return err
	}

	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, v1.Node{})
	if err != nil {
		return err
	}

	if _, err := client.CoreV1().Nodes().Patch(node.Name, types.StrategicMergePatchType, patchBytes); err != nil {
		if apierrs.IsConflict(err) {
			fmt.Println("Temporarily unable to update node metadata due to conflict (will retry)")
		}
		return err
	}

	return nil
}

/*
Copyright 2016 The Kubernetes Authors.

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

package cmd

import (
	"fmt"
	"io"
	"strings"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/fields"
	cmdutil "k8s.io/kubernetes/pkg/kubectl/cmd/util"
	"k8s.io/kubernetes/pkg/runtime"
)

var (
	maintenanceExample = dedent.Dedent(`
		# Turn on maintenance mode for node1.
		kubectl maintenance node1 on
		# Turn off maintenance mode for node1.
		kubectl maintenance node1 off
		`)
)

func NewCmdMaintenance(f *cmdutil.Factory, out io.Writer) *cobra.Command {
	cmd := &cobra.Command{
		Use:     "maintenance NODE on/off",
		Short:   "Prepare node for running maintenance",
		Example: maintenanceExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(runMaintenance(args[0], args[1], f, out))
		},
	}
	return cmd
}

func runMaintenance(nodeName string, switchName string, f *cmdutil.Factory, out io.Writer) error {
	client, err := f.ClientSet()
	if err != nil {
		return err
	}

	var ms maintenanceSwitch
	switch strings.ToLower(switchName) {
	case "on":
		ms = maintenanceSwitch{
			nodeUpdateMessage: "Updated node %s with maintenance:NoSchedule taint\n",
			podUpdateMessage:  "Updated pod %v:%v with %v toleration\n",
			modifyTaints:      api.AddTaints,
			modifyTolerations: api.AddTolerations,
		}

	case "off":
		ms = maintenanceSwitch{
			nodeUpdateMessage: "Removed from node %s maintenance:NoSchedule taint\n",
			podUpdateMessage:  "Removed from pod %v:%v nodeoutage:NoExecute toleration\n",
			modifyTaints:      api.RemoveTaints,
			modifyTolerations: api.RemoveTolerations,
		}

	default:
		return fmt.Errorf("Unknown switch %v, please use ON/OFF", switchName)
	}
	if err := ms.Switch(client, out, nodeName); err != nil {
		return err
	}
	return nil
}

type maintenanceSwitch struct {
	podUpdateMessage  string
	nodeUpdateMessage string
	modifyTaints      func(obj runtime.Object, taints ...api.Taint) (bool, error)
	modifyTolerations func(obj runtime.Object, tolerations ...api.Toleration) (bool, error)
}

func (ms maintenanceSwitch) Switch(client *internalclientset.Clientset, out io.Writer, nodeName string) error {
	node, err := client.Nodes().Get(nodeName)
	if err != nil {
		return err
	}
	maintenanceTaint := api.Taint{Key: unversioned.TaintNodeMaintenance, Effect: api.TaintEffectNoSchedule}
	nodeoutageToleration := api.Toleration{Key: unversioned.TaintNodeOutage, Operator: api.TolerationOpEqual, Effect: api.TaintEffectNoExecute}
	updated, err := ms.modifyTaints(node, maintenanceTaint)
	if err != nil {
		return err
	}
	if updated {
		if _, err := client.Nodes().Update(node); err != nil {
			return err
		}
		fmt.Fprintf(out, ms.nodeUpdateMessage, node.Name)
	}

	podList, err := client.Core().Pods(api.NamespaceAll).List(api.ListOptions{
		FieldSelector: fields.SelectorFromSet(fields.Set{"spec.nodeName": node.Name})})
	if err != nil {
		return err
	}

	// TODO aggregate and handle errors
	for _, pod := range podList.Items {
		updated, err := ms.modifyTolerations(&pod, nodeoutageToleration)
		if err == nil && updated {
			client.Core().Pods(pod.Namespace).Update(&pod)
			fmt.Fprintf(out, ms.podUpdateMessage, pod.Namespace, pod.Name, nodeoutageToleration)
		}
	}
	return nil
}

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
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
)

type controlPlanePrepareData interface {
	Cfg() *kubeadmapi.JoinConfiguration
	InitCfg() (*kubeadmapi.InitConfiguration, error)
}

// NewControlPlanePreparePhase creates a kubeadm workflow phase that implements the preparation of the node to serve a control plane
func NewControlPlanePreparePhase() workflow.Phase {
	return workflow.Phase{
		Name:  "control-plane-prepare",
		Short: "Prepares the machine for serving a control plane.",
		Long:  cmdutil.MacroCommandLongDescription,
		Phases: []workflow.Phase{
			{
				Name:           "all",
				Short:          "Prepares the machine for serving a control plane.",
				InheritFlags:   getControlPlanePreparePhaseFlags(),
				RunAllSiblings: true,
			},
			newControlPlanePrepareCertsSubphase(),
			newControlPlanePrepareKubeconfigSubphase(),
			newControlPlanePrepareManifestsSubphases(),
		},
	}
}

func getControlPlanePreparePhaseFlags() []string {
	return []string{
		options.APIServerAdvertiseAddress,
		options.APIServerBindPort,
		options.CfgPath,
		options.ControlPlane,
		options.NodeName,
		options.TokenDiscovery,
		options.TokenDiscoveryCAHash,
		options.TokenDiscoverySkipCAHash,
	}
}

func newControlPlanePrepareCertsSubphase() workflow.Phase {
	return workflow.Phase{
		Name:         "certs",
		Short:        "Generates the certificates for the new control plane components",
		Run:          runControlPlanePrepareCertsPhaseLocal,
		InheritFlags: getControlPlanePreparePhaseFlags(), //NB. eventually in future we would like to break down this in sub phases for each cert or add the --csr option
	}
}

func newControlPlanePrepareKubeconfigSubphase() workflow.Phase {
	return workflow.Phase{
		Name:         "kubeconfig",
		Short:        "Generates the kubeconfig for the new control plane components",
		Run:          runControlPlanePrepareKubeconfigPhaseLocal,
		InheritFlags: getControlPlanePreparePhaseFlags(), //NB. eventually in future we would like to break down this in sub phases for each kubeconfig
	}
}

func newControlPlanePrepareManifestsSubphases() workflow.Phase {
	return workflow.Phase{
		Name:         "manifests",
		Short:        "Generates the manifests for the new control plane components",
		Run:          runControlPlanePrepareManifestsSubphase,
		InheritFlags: getControlPlanePreparePhaseFlags(), //NB. eventually in future we would like to break down this in sub phases for each component
	}
}

func runControlPlanePrepareManifestsSubphase(c workflow.RunData) error {
	data, ok := c.(controlPlanePrepareData)
	if !ok {
		return errors.New("control-plane-prepare phase invoked with an invalid data struct")
	}

	// Skip if this is not a control plane
	if data.Cfg().ControlPlane == nil {
		return nil
	}

	cfg, err := data.InitCfg()
	if err != nil {
		return err
	}

	// Generate missing certificates (if any)
	return controlplane.CreateInitStaticPodManifestFiles(kubeadmconstants.GetStaticPodDirectory(), cfg)
}

func runControlPlanePrepareCertsPhaseLocal(c workflow.RunData) error {
	data, ok := c.(controlPlanePrepareData)
	if !ok {
		return errors.New("control-plane-prepare phase invoked with an invalid data struct")
	}

	// Skip if this is not a control plane
	if data.Cfg().ControlPlane == nil {
		return nil
	}

	cfg, err := data.InitCfg()
	if err != nil {
		return err
	}

	// Generate missing certificates (if any)
	return certsphase.CreatePKIAssets(cfg)
}

func runControlPlanePrepareKubeconfigPhaseLocal(c workflow.RunData) error {
	data, ok := c.(controlPlanePrepareData)
	if !ok {
		return errors.New("control-plane-prepare phase invoked with an invalid data struct")
	}

	// Skip if this is not a control plane
	if data.Cfg().ControlPlane == nil {
		return nil
	}

	cfg, err := data.InitCfg()
	if err != nil {
		return err
	}

	fmt.Println("[control-plane-prepare] Generating kubeconfig files")

	// Generate kubeconfig files for controller manager, scheduler and for the admin/kubeadm itself
	// NB. The kubeconfig file for kubelet will be generated by the TLS bootstrap process in
	// following steps of the join --experimental-control plane workflow
	if err := kubeconfigphase.CreateJoinControlPlaneKubeConfigFiles(kubeadmconstants.KubernetesDir, cfg); err != nil {
		return errors.Wrap(err, "error generating kubeconfig files")
	}

	return nil
}

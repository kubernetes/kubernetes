/*
Copyright 2019 The Kubernetes Authors.

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
	clientset "k8s.io/client-go/kubernetes"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	etcdphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd"
	markcontrolplanephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markcontrolplane"
	uploadconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/uploadconfig"
	"k8s.io/kubernetes/pkg/util/normalizer"
)

var (
	controlPlaneJoinExample = normalizer.Examples(`
		# Joins a machine as a control plane instance
		kubeadm join phase control-plane-join all
		`)
)

type controlPlaneJoinData interface {
	Cfg() *kubeadmapi.JoinConfiguration
	ClientSetFromFile(string) (*clientset.Clientset, error)
	InitCfg() (*kubeadmapi.InitConfiguration, error)
	KubeConfigPath() string
}

func getControlPlaneJoinPhaseFlags() []string {
	return []string{
		options.APIServerAdvertiseAddress,
		options.APIServerBindPort,
		options.CfgPath,
		options.ControlPlane,
		options.NodeName,
		options.TokenDiscovery,
		options.TokenDiscoveryCAHash,
		options.TokenDiscoverySkipCAHash,
		options.KubeconfigPath,
	}
}

// NewControlPlaneJoinPhase creates a kubeadm workflow phase that implements joining a machine as a control plane instance
func NewControlPlaneJoinPhase() workflow.Phase {
	return workflow.Phase{
		Name:    "control-plane-join",
		Short:   "Joins a machine as a control plane instance",
		Long:    cmdutil.MacroCommandLongDescription,
		Example: controlPlaneJoinExample,
		Phases: []workflow.Phase{
			{
				Name:           "all",
				Short:          "Joins a machine as a control plane instance",
				InheritFlags:   getControlPlaneJoinPhaseFlags(),
				RunAllSiblings: true,
			},
			newEtcdLocalSubphase(),
			newUploadConfigSubphase(),
			newMarkControlPlaneSubphase(),
		},
	}
}

func newEtcdLocalSubphase() workflow.Phase {
	return workflow.Phase{
		Name:         "etcd",
		Short:        "Generates the static Pod manifest file for a local etcd member",
		Run:          runEtcdPhase,
		InheritFlags: getControlPlaneJoinPhaseFlags(),
	}
}

func newUploadConfigSubphase() workflow.Phase {
	return workflow.Phase{
		Name:         "upload-config",
		Short:        "Upload the currently used configuration to the cluster",
		Run:          runUploadConfigPhase,
		InheritFlags: getControlPlaneJoinPhaseFlags(),
	}
}

func newMarkControlPlaneSubphase() workflow.Phase {
	return workflow.Phase{
		Name:         "mark-control-plane",
		Short:        "Mark a node as a control-plane",
		Run:          runMarkControlPlanePhase,
		InheritFlags: getControlPlaneJoinPhaseFlags(),
	}
}

func runEtcdPhase(c workflow.RunData) error {
	data, ok := c.(controlPlaneJoinData)
	if !ok {
		return errors.New("control-plane-join phase invoked with an invalid data struct")
	}

	if data.Cfg().ControlPlane == nil {
		return nil
	}

	kubeConfigFile := data.KubeConfigPath()

	client, err := data.ClientSetFromFile(kubeConfigFile)
	if err != nil {
		return errors.Wrap(err, "couldn't create Kubernetes client")
	}
	cfg, err := data.InitCfg()
	if err != nil {
		return err
	}
	// in case of local etcd
	if cfg.Etcd.External != nil {
		fmt.Println("[control-plane-join] using external etcd - no local stacked instance added")
		return nil
	}

	// Adds a new etcd instance; in order to do this the new etcd instance should be "announced" to
	// the existing etcd members before being created.
	// This operation must be executed after kubelet is already started in order to minimize the time
	// between the new etcd member is announced and the start of the static pod running the new etcd member, because during
	// this time frame etcd gets temporary not available (only when moving from 1 to 2 members in the etcd cluster).
	// From https://coreos.com/etcd/docs/latest/v2/runtime-configuration.html
	// "If you add a new member to a 1-node cluster, the cluster cannot make progress before the new member starts
	// because it needs two members as majority to agree on the consensus. You will only see this behavior between the time
	// etcdctl member add informs the cluster about the new member and the new member successfully establishing a connection to the 	// existing one."
	if err := etcdphase.CreateStackedEtcdStaticPodManifestFile(client, kubeadmconstants.GetStaticPodDirectory(), cfg.NodeRegistration.Name, &cfg.ClusterConfiguration, &cfg.LocalAPIEndpoint); err != nil {
		return errors.Wrap(err, "error creating local etcd static pod manifest file")
	}

	return nil
}

func runUploadConfigPhase(c workflow.RunData) error {
	data, ok := c.(controlPlaneJoinData)
	if !ok {
		return errors.New("control-plane-join phase invoked with an invalid data struct")
	}

	if data.Cfg().ControlPlane == nil {
		return nil
	}

	kubeConfigFile := data.KubeConfigPath()

	client, err := data.ClientSetFromFile(kubeConfigFile)
	if err != nil {
		return errors.Wrap(err, "couldn't create Kubernetes client")
	}
	cfg, err := data.InitCfg()
	if err != nil {
		return err
	}

	if err := uploadconfigphase.UploadConfiguration(cfg, client); err != nil {
		return errors.Wrap(err, "error uploading configuration")
	}

	return nil
}

func runMarkControlPlanePhase(c workflow.RunData) error {
	data, ok := c.(controlPlaneJoinData)
	if !ok {
		return errors.New("control-plane-join phase invoked with an invalid data struct")
	}

	if data.Cfg().ControlPlane == nil {
		return nil
	}

	kubeConfigFile := data.KubeConfigPath()

	client, err := data.ClientSetFromFile(kubeConfigFile)
	if err != nil {
		return errors.Wrap(err, "couldn't create Kubernetes client")
	}
	cfg, err := data.InitCfg()
	if err != nil {
		return err
	}

	if err := markcontrolplanephase.MarkControlPlane(client, cfg.NodeRegistration.Name, cfg.NodeRegistration.Taints); err != nil {
		return errors.Wrap(err, "error applying control-plane label and taints")
	}

	return nil
}

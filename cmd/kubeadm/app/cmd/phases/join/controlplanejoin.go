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

	"github.com/spf13/cobra"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	cmdutil "k8s.io/kubernetes/cmd/kubeadm/app/cmd/util"
	etcdphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd"
	markcontrolplanephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/markcontrolplane"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	etcdutil "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
)

var controlPlaneJoinExample = cmdutil.Examples(`
	# Joins a machine as a control plane instance
	kubeadm join phase control-plane-join all
`)

var etcdJoinExample = cmdutil.Examples(`
	# Joins etcd for a control plane instance
	kubeadm join phase control-plane-join-etcd all
`)

func getControlPlaneJoinPhaseFlags(name string) []string {
	flags := []string{
		options.CfgPath,
		options.ControlPlane,
		options.NodeName,
	}
	if name == "etcd" || name == "all" {
		flags = append(flags, options.Patches)
	}
	if name != "mark-control-plane" {
		flags = append(flags, options.APIServerAdvertiseAddress)
	}
	if name != "update-status" {
		flags = append(flags, options.DryRun)
	}
	return flags
}

// NewControlPlaneJoinPhase creates a kubeadm workflow phase that implements joining a machine as a control plane instance
func NewControlPlaneJoinPhase() workflow.Phase {
	return workflow.Phase{
		Name:    "control-plane-join",
		Short:   "Join a machine as a control plane instance",
		Example: controlPlaneJoinExample,
		Phases: []workflow.Phase{
			{
				Name:           "all",
				Short:          "Join a machine as a control plane instance",
				InheritFlags:   getControlPlaneJoinPhaseFlags("all"),
				RunAllSiblings: true,
				ArgsValidator:  cobra.NoArgs,
			},
			newMarkControlPlaneSubphase(),
		},
	}
}

// NewEtcdJoinPhase creates a kubeadm workflow phase that implements joining etcd
func NewEtcdJoinPhase() workflow.Phase {
	return workflow.Phase{
		Name:          "etcd-join",
		Short:         "Join etcd for control plane nodes",
		Run:           runEtcdPhase,
		Example:       etcdJoinExample,
		InheritFlags:  getControlPlaneJoinPhaseFlags("etcd"),
		ArgsValidator: cobra.NoArgs,
	}
}

func newMarkControlPlaneSubphase() workflow.Phase {
	return workflow.Phase{
		Name:          "mark-control-plane",
		Short:         "Mark a node as a control-plane",
		Run:           runMarkControlPlanePhase,
		InheritFlags:  getControlPlaneJoinPhaseFlags("mark-control-plane"),
		ArgsValidator: cobra.NoArgs,
	}
}

func runEtcdPhase(c workflow.RunData) error {
	data, ok := c.(JoinData)
	if !ok {
		return errors.New("control-plane-join phase invoked with an invalid data struct")
	}

	if data.Cfg().ControlPlane == nil {
		return nil
	}

	// gets access to the cluster using the identity defined in admin.conf
	client, err := data.Client()
	if err != nil {
		return errors.Wrap(err, "couldn't create Kubernetes client")
	}
	cfg, err := data.InitCfg()
	if err != nil {
		return err
	}
	// in case of external etcd
	if cfg.Etcd.External != nil {
		fmt.Println("[control-plane-join] Using external etcd - no local stacked instance added")
		return nil
	}

	if !data.DryRun() {
		// Create the etcd data directory
		if err := etcdutil.CreateDataDirectory(cfg.Etcd.Local.DataDir); err != nil {
			return err
		}
	} else {
		fmt.Printf("[control-plane-join] Would ensure that %q directory is present\n", cfg.Etcd.Local.DataDir)
	}

	// Adds a new etcd instance; in order to do this the new etcd instance should be "announced" to
	// the existing etcd members before being created.
	// This operation must be executed after kubelet is already started in order to minimize the time
	// between the new etcd member is announced and the start of the static pod running the new etcd member, because during
	// this time frame etcd gets temporary not available (only when moving from 1 to 2 members in the etcd cluster).
	// From https://coreos.com/etcd/docs/latest/v2/runtime-configuration.html
	// "If you add a new member to a 1-node cluster, the cluster cannot make progress before the new member starts
	// because it needs two members as majority to agree on the consensus. You will only see this behavior between the time
	// etcdctl member add informs the cluster about the new member and the new member successfully establishing a connection to the
	// existing one."
	if err := etcdphase.CreateStackedEtcdStaticPodManifestFile(client, data.ManifestDir(), data.PatchesDir(), cfg.NodeRegistration.Name, &cfg.ClusterConfiguration, &cfg.LocalAPIEndpoint, data.DryRun(), data.CertificateWriteDir()); err != nil {
		return errors.Wrap(err, "error creating local etcd static pod manifest file")
	}

	return nil
}

func runMarkControlPlanePhase(c workflow.RunData) error {
	data, ok := c.(JoinData)
	if !ok {
		return errors.New("control-plane-join phase invoked with an invalid data struct")
	}

	if data.Cfg().ControlPlane == nil {
		return nil
	}

	// gets access to the cluster using the identity defined in admin.conf
	client, err := data.Client()
	if err != nil {
		return errors.Wrap(err, "couldn't create Kubernetes client")
	}
	cfg, err := data.InitCfg()
	if err != nil {
		return err
	}

	if !data.DryRun() {
		if err := markcontrolplanephase.MarkControlPlane(client, cfg.NodeRegistration.Name, cfg.NodeRegistration.Taints); err != nil {
			return errors.Wrap(err, "error applying control-plane label and taints")
		}
	} else {
		fmt.Printf("[control-plane-join] Would mark node %s as a control-plane\n", cfg.NodeRegistration.Name)
	}

	return nil
}

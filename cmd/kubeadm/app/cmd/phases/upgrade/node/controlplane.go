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

package node

import (
	"fmt"
	"os"

	"github.com/pkg/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// NewControlPlane creates a kubeadm workflow phase that implements handling of control-plane upgrade.
func NewControlPlane() workflow.Phase {
	phase := workflow.Phase{
		Name:  "control-plane",
		Short: "Upgrade the control plane instance deployed on this node, if any",
		Run:   runControlPlane(),
		InheritFlags: []string{
			options.DryRun,
			options.KubeconfigPath,
			options.CertificateRenewal,
			options.EtcdUpgrade,
			options.Kustomize,
		},
	}
	return phase
}

func runControlPlane() func(c workflow.RunData) error {
	return func(c workflow.RunData) error {
		data, ok := c.(Data)
		if !ok {
			return errors.New("control-plane phase invoked with an invalid data struct")
		}

		// if this is not a control-plane node, this phase should not be executed
		if !data.IsControlPlaneNode() {
			fmt.Println("[upgrade] Skipping phase. Not a control plane node.")
			return nil
		}

		// otherwise, retrieve all the info required for control plane upgrade
		cfg := data.Cfg()
		client := data.Client()
		dryRun := data.DryRun()
		etcdUpgrade := data.EtcdUpgrade()
		renewCerts := data.RenewCerts()
		kustomizeDir := data.KustomizeDir()

		// Upgrade the control plane and etcd if installed on this node
		fmt.Printf("[upgrade] Upgrading your Static Pod-hosted control plane instance to version %q...\n", cfg.KubernetesVersion)
		if dryRun {
			return upgrade.DryRunStaticPodUpgrade(kustomizeDir, cfg)
		}

		waiter := apiclient.NewKubeWaiter(data.Client(), upgrade.UpgradeManifestTimeout, os.Stdout)

		if err := upgrade.PerformStaticPodUpgrade(client, waiter, cfg, etcdUpgrade, renewCerts, kustomizeDir); err != nil {
			return errors.Wrap(err, "couldn't complete the static pod upgrade")
		}

		fmt.Println("[upgrade] The control plane instance for this node was successfully updated!")

		return nil
	}
}

/*
Copyright 2024 The Kubernetes Authors.

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

// Package apply implements phases of 'kubeadm upgrade apply'.
package apply

import (
	"fmt"
	"os"

	"github.com/pkg/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/upgrade"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
)

// NewControlPlanePhase returns a new control-plane phase.
func NewControlPlanePhase() workflow.Phase {
	phase := workflow.Phase{
		Name:  "control-plane",
		Short: "Upgrade the control plane",
		Run:   runControlPlane,
		InheritFlags: []string{
			options.CfgPath,
			options.KubeconfigPath,
			options.DryRun,
			options.CertificateRenewal,
			options.EtcdUpgrade,
			options.Patches,
		},
	}
	return phase
}

func runControlPlane(c workflow.RunData) error {
	data, ok := c.(Data)
	if !ok {
		return errors.New("control-plane phase invoked with an invalid data struct")
	}

	initCfg, upgradeCfg, client, patchesDir := data.InitCfg(), data.Cfg(), data.Client(), data.PatchesDir()

	if data.DryRun() {
		fmt.Printf("[dryrun] Would upgrade your static Pod-hosted control plane to version %q", initCfg.KubernetesVersion)
		return upgrade.DryRunStaticPodUpgrade(patchesDir, initCfg)
	}

	fmt.Printf("[upgrade/control-plane] Upgrading your static Pod-hosted control plane to version %q (timeout: %v)...\n",
		initCfg.KubernetesVersion, upgradeCfg.Timeouts.UpgradeManifests.Duration)

	waiter := apiclient.NewKubeWaiter(client, upgradeCfg.Timeouts.UpgradeManifests.Duration, os.Stdout)
	if err := upgrade.PerformStaticPodUpgrade(client, waiter, initCfg, data.EtcdUpgrade(), data.RenewCerts(), patchesDir); err != nil {
		return errors.Wrap(err, "couldn't complete the static pod upgrade")
	}

	fmt.Println("[upgrade/control-plane] The control plane instance for this node was successfully upgraded!")

	return nil
}

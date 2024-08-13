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

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	etcdphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd"
)

// NewCheckEtcdPhase is a hidden phase that runs after the control-plane-prepare and
// before the bootstrap-kubelet phase that ensures etcd is healthy
func NewCheckEtcdPhase() workflow.Phase {
	return workflow.Phase{
		Name:   "check-etcd",
		Run:    runCheckEtcdPhase,
		Hidden: true,
	}
}

func runCheckEtcdPhase(c workflow.RunData) error {
	data, ok := c.(JoinData)
	if !ok {
		return errors.New("check-etcd phase invoked with an invalid data struct")
	}

	// Skip if this is not a control plane
	if data.Cfg().ControlPlane == nil {
		return nil
	}

	cfg, err := data.InitCfg()
	if err != nil {
		return err
	}

	if cfg.Etcd.External != nil {
		fmt.Println("[check-etcd] Skipping etcd check in external mode")
		return nil
	}

	fmt.Println("[check-etcd] Checking that the etcd cluster is healthy")

	// Checks that the etcd cluster is healthy
	// NB. this check cannot be implemented before because it requires the admin.conf and all the certificates
	//     for connecting to etcd already in place
	client, err := data.Client()
	if err != nil {
		return err
	}

	return etcdphase.CheckLocalEtcdClusterStatus(client, data.CertificateWriteDir())
}

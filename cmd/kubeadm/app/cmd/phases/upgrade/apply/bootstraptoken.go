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

	errorsutil "k8s.io/apimachinery/pkg/util/errors"

	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/options"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/workflow"
	clusterinfophase "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/clusterinfo"
	nodebootstraptoken "k8s.io/kubernetes/cmd/kubeadm/app/phases/bootstraptoken/node"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

// NewBootstrapTokenPhase returns a new bootstrap-token phase.
func NewBootstrapTokenPhase() workflow.Phase {
	return workflow.Phase{
		Name:  "bootstrap-token",
		Short: "Configures bootstrap token and cluster-info RBAC rules",
		InheritFlags: []string{
			options.CfgPath,
			options.KubeconfigPath,
			options.DryRun,
		},
		Run: runBootstrapToken,
	}
}

func runBootstrapToken(c workflow.RunData) error {
	data, ok := c.(Data)
	if !ok {
		return errors.New("bootstrap-token phase invoked with an invalid data struct")
	}

	if data.DryRun() {
		fmt.Println("[dryrun] Would configure bootstrap token and cluster-info RBAC rules")
		return nil
	}

	fmt.Println("[upgrade/bootstrap-token] Configuring bootstrap token and cluster-info RBAC rules")

	client := data.Client()

	var errs []error
	// Create RBAC rules that makes the bootstrap tokens able to get nodes
	if err := nodebootstraptoken.AllowBootstrapTokensToGetNodes(client); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the bootstrap tokens able to post CSRs
	if err := nodebootstraptoken.AllowBootstrapTokensToPostCSRs(client); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the bootstrap tokens able to get their CSRs approved automatically
	if err := nodebootstraptoken.AutoApproveNodeBootstrapTokens(client); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the nodes to rotate certificates and get their CSRs approved automatically
	if err := nodebootstraptoken.AutoApproveNodeCertificateRotation(client); err != nil {
		errs = append(errs, err)
	}

	// Create/update RBAC rules that makes the cluster-info ConfigMap reachable
	if err := clusterinfophase.CreateClusterInfoRBACRules(client); err != nil {
		errs = append(errs, err)
	}

	return errorsutil.NewAggregate(errs)
}

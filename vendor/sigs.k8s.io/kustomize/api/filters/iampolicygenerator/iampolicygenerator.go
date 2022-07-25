// Copyright 2021 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package iampolicygenerator

import (
	"fmt"

	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

type Filter struct {
	IAMPolicyGenerator types.IAMPolicyGeneratorArgs `json:",inline,omitempty" yaml:",inline,omitempty"`
}

// Filter adds a GKE service account object to nodes
func (f Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	switch f.IAMPolicyGenerator.Cloud {
	case types.GKE:
		IAMPolicyResources, err := f.generateGkeIAMPolicyResources()
		if err != nil {
			return nil, err
		}
		nodes = append(nodes, IAMPolicyResources...)
	default:
		return nil, fmt.Errorf("cloud provider %s not supported yet", f.IAMPolicyGenerator.Cloud)
	}
	return nodes, nil
}

func (f Filter) generateGkeIAMPolicyResources() ([]*yaml.RNode, error) {
	var result []*yaml.RNode
	input := fmt.Sprintf(`
apiVersion: v1
kind: ServiceAccount
metadata:
  annotations:
    iam.gke.io/gcp-service-account: %s@%s.iam.gserviceaccount.com
  name: %s
`, f.IAMPolicyGenerator.ServiceAccount.Name,
		f.IAMPolicyGenerator.ProjectId,
		f.IAMPolicyGenerator.KubernetesService.Name)

	if f.IAMPolicyGenerator.Namespace != "" {
		input = input + fmt.Sprintf("\n  namespace: %s", f.IAMPolicyGenerator.Namespace)
	}

	sa, err := yaml.Parse(input)
	if err != nil {
		return nil, err
	}

	return append(result, sa), nil
}

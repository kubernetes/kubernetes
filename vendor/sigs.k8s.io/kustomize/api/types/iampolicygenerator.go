// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

type Cloud string

const GKE Cloud = "gke"

// IAMPolicyGeneratorArgs contains arguments to generate a GKE service account resource.
type IAMPolicyGeneratorArgs struct {
	// which cloud provider to generate for (e.g. "gke")
	Cloud `json:"cloud" yaml:"cloud"`

	// information about the kubernetes cluster for this object
	KubernetesService `json:"kubernetesService" yaml:"kubernetesService"`

	// information about the service account and project
	ServiceAccount `json:"serviceAccount" yaml:"serviceAccount"`
}

type KubernetesService struct {
	// the name used for the Kubernetes service account
	Name string `json:"name" yaml:"name"`

	// the name of the Kubernetes namespace for this object
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`
}

type ServiceAccount struct {
	// the name of the new cloud provider service account
	Name string `json:"name" yaml:"name"`

	// The ID of the project
	ProjectId string `json:"projectId" yaml:"projectId"`
}

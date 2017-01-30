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

package kubectl

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/pkg/api/v1"
)

// ClusterGeneratorV1Beta1 supports stable generation of a
// federation/cluster resource.
type ClusterGeneratorV1Beta1 struct {
	// Name of the cluster context (required)
	Name string
	// ClientCIDR is the CIDR range in which the Kubernetes APIServer
	// is available for the client (optional)
	ClientCIDR string
	// ServerAddress is the APIServer address of the Kubernetes cluster
	// that is being registered (required)
	ServerAddress string
	// SecretName is the name of the secret that stores the credentials
	// for the Kubernetes cluster that is being registered (optional)
	SecretName string
}

// Ensure it supports the generator pattern that uses parameter
// injection.
var _ Generator = &ClusterGeneratorV1Beta1{}

// Ensure it supports the generator pattern that uses parameters
// specified during construction.
var _ StructuredGenerator = &ClusterGeneratorV1Beta1{}

// Generate returns a cluster resource using the specified parameters.
func (s ClusterGeneratorV1Beta1) Generate(genericParams map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), genericParams)
	if err != nil {
		return nil, err
	}
	clustergen := &ClusterGeneratorV1Beta1{}
	params := map[string]string{}
	for key, value := range genericParams {
		strVal, isString := value.(string)
		if !isString {
			return nil, fmt.Errorf("expected string, saw %v for '%s'", value, key)
		}
		params[key] = strVal
	}
	clustergen.Name = params["name"]
	clustergen.ClientCIDR = params["client-cidr"]
	clustergen.ServerAddress = params["server-address"]
	clustergen.SecretName = params["secret"]
	return clustergen.StructuredGenerate()
}

// ParamNames returns the set of supported input parameters when using
// the parameter injection generator pattern.
func (s ClusterGeneratorV1Beta1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"client-cidr", false},
		{"server-address", true},
		{"secret", false},
	}
}

// StructuredGenerate outputs a federation cluster resource object
// using the configured fields.
func (s ClusterGeneratorV1Beta1) StructuredGenerate() (runtime.Object, error) {
	if err := s.validate(); err != nil {
		return nil, err
	}
	if s.ClientCIDR == "" {
		s.ClientCIDR = "0.0.0.0/0"
	}
	if s.SecretName == "" {
		s.SecretName = s.Name
	}
	cluster := &federationapi.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name: s.Name,
		},
		Spec: federationapi.ClusterSpec{
			ServerAddressByClientCIDRs: []federationapi.ServerAddressByClientCIDR{
				{
					ClientCIDR:    s.ClientCIDR,
					ServerAddress: s.ServerAddress,
				},
			},
			SecretRef: &v1.LocalObjectReference{
				Name: s.SecretName,
			},
		},
	}
	return cluster, nil
}

// validate validates required fields are set to support structured
// generation.
func (s ClusterGeneratorV1Beta1) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if len(s.ServerAddress) == 0 {
		return fmt.Errorf("server address must be specified")
	}
	return nil
}

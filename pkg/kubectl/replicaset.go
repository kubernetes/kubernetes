/*
Copyright 2017 The Kubernetes Authors.

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

	appsv1beta2 "k8s.io/api/apps/v1beta2"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// BaseReplicaSetGenerator: implement the common functionality of
// ReplicaSetGeneratorAppsV1Beta2. To reduce confusion, it's best to
// keep this struct in the same file as those generators.
type BaseReplicaSetGenerator struct {
	Name     string
	Images   []string
	Replicas int32
}

// validate: check if the caller has forgotten to set one of our fields.
func (b BaseReplicaSetGenerator) validate() error {
	if len(b.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if len(b.Images) == 0 {
		return fmt.Errorf("at least one image must be specified")
	}
	if b.Replicas < 1 {
		return fmt.Errorf("replicas must not be less than 1")
	}
	return nil
}

// structuredGenerate: determine the fields of a replicaset. The struct that
// embeds BaseReplicaSetGenerator should assemble these pieces into a
// runtime.Object.
func (b BaseReplicaSetGenerator) structuredGenerate() (podSpec v1.PodSpec, labels map[string]string, selector metav1.LabelSelector, err error) {
	err = b.validate()
	if err != nil {
		return
	}
	podSpec = buildPodSpec(b.Images)
	labels = map[string]string{}
	labels["app"] = b.Name
	selector = metav1.LabelSelector{MatchLabels: labels}
	return
}

// ReplicaSetGeneratorAppsV1Beta2 supports stable generation of a replicaset
type ReplicaSetGeneratorAppsV1Beta2 struct {
	BaseReplicaSetGenerator
}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ StructuredGenerator = &ReplicaSetGeneratorAppsV1Beta2{}

// StructuredGenerate outputs a replicaset object using the configured fields
func (s *ReplicaSetGeneratorAppsV1Beta2) StructuredGenerate() (runtime.Object, error) {
	podSpec, labels, selector, err := s.structuredGenerate()
	return &appsv1beta2.ReplicaSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:   s.Name,
			Labels: labels,
		},
		Spec: appsv1beta2.ReplicaSetSpec{
			Replicas: &s.Replicas,
			Selector: &selector,
			Template: v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: podSpec,
			},
		},
	}, err
}

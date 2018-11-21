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

package versioned

import (
	"fmt"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	appsv1beta1 "k8s.io/api/apps/v1beta1"
	"k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/kubectl/generate"
)

// BaseDeploymentGenerator implements the common functionality of
// DeploymentBasicGeneratorV1, DeploymentBasicAppsGeneratorV1Beta1 and DeploymentBasicAppsGeneratorV1. To reduce
// confusion, it's best to keep this struct in the same file as those
// generators.
type BaseDeploymentGenerator struct {
	Name   string
	Images []string
}

// validate: check if the caller has forgotten to set one of our fields.
func (b BaseDeploymentGenerator) validate() error {
	if len(b.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if len(b.Images) == 0 {
		return fmt.Errorf("at least one image must be specified")
	}
	return nil
}

// structuredGenerate: determine the fields of a deployment. The struct that
// embeds BaseDeploymentGenerator should assemble these pieces into a
// runtime.Object.
func (b BaseDeploymentGenerator) structuredGenerate() (
	podSpec v1.PodSpec,
	labels map[string]string,
	selector metav1.LabelSelector,
	err error,
) {
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

// buildPodSpec: parse the image strings and assemble them into the Containers
// of a PodSpec. This is all you need to create the PodSpec for a deployment.
func buildPodSpec(images []string) v1.PodSpec {
	podSpec := v1.PodSpec{Containers: []v1.Container{}}
	for _, imageString := range images {
		// Retain just the image name
		imageSplit := strings.Split(imageString, "/")
		name := imageSplit[len(imageSplit)-1]
		// Remove any tag or hash
		if strings.Contains(name, ":") {
			name = strings.Split(name, ":")[0]
		}
		if strings.Contains(name, "@") {
			name = strings.Split(name, "@")[0]
		}
		podSpec.Containers = append(podSpec.Containers, v1.Container{Name: name, Image: imageString})
	}
	return podSpec
}

// DeploymentBasicGeneratorV1 supports stable generation of a deployment
type DeploymentBasicGeneratorV1 struct {
	BaseDeploymentGenerator
}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ generate.StructuredGenerator = &DeploymentBasicGeneratorV1{}

// StructuredGenerate outputs a deployment object using the configured fields
func (s *DeploymentBasicGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	podSpec, labels, selector, err := s.structuredGenerate()
	one := int32(1)
	return &extensionsv1beta1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   s.Name,
			Labels: labels,
		},
		Spec: extensionsv1beta1.DeploymentSpec{
			Replicas: &one,
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

// DeploymentBasicAppsGeneratorV1Beta1 supports stable generation of a deployment under apps/v1beta1 endpoint
type DeploymentBasicAppsGeneratorV1Beta1 struct {
	BaseDeploymentGenerator
}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ generate.StructuredGenerator = &DeploymentBasicAppsGeneratorV1Beta1{}

// StructuredGenerate outputs a deployment object using the configured fields
func (s *DeploymentBasicAppsGeneratorV1Beta1) StructuredGenerate() (runtime.Object, error) {
	podSpec, labels, selector, err := s.structuredGenerate()
	one := int32(1)
	return &appsv1beta1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   s.Name,
			Labels: labels,
		},
		Spec: appsv1beta1.DeploymentSpec{
			Replicas: &one,
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

// DeploymentBasicAppsGeneratorV1 supports stable generation of a deployment under apps/v1 endpoint
type DeploymentBasicAppsGeneratorV1 struct {
	BaseDeploymentGenerator
}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ generate.StructuredGenerator = &DeploymentBasicAppsGeneratorV1{}

// StructuredGenerate outputs a deployment object using the configured fields
func (s *DeploymentBasicAppsGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	podSpec, labels, selector, err := s.structuredGenerate()
	one := int32(1)
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   s.Name,
			Labels: labels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &one,
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

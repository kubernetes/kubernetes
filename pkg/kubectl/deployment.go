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
	"strings"

	appsv1beta1 "k8s.io/api/apps/v1beta1"
	"k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// BaseDeploymentGenerator: implement the common functionality of
// DeploymentBasicGeneratorV1 and DeploymentBasicAppsGeneratorV1 (both of the
// 'kubectl create deployment' Generators). To reduce confusion, it's best to
// keep this struct in the same file as those generators.
type BaseDeploymentGenerator struct {
	Name   string
	Images []string

	// Replicas is not optional in this struct but typically it defaults to
	// 1 in the command system.
	// Determines the number of replicas on the deployment.
	Replicas int32

	// Limits and Requests are strings like "cpu=200m,memory=512Mi".
	Limits   string
	Requests string

	// Command is validated in a special way. If the command is specified,
	// there must be exactly 1 image in the Images slice.
	Command []string
	Args    []string
}

// validate: check if the caller has forgotten to set one of our fields.
// We don't bother to check if the optional fields have been set. Do not add
// validation to the optional fields if it something that can be caught at a
// lower level and bubbled up. BaseDeploymentGenerator is just a way to get
// parameters into a Generator.
func (b BaseDeploymentGenerator) validate() error {
	if len(b.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if len(b.Images) == 0 {
		return fmt.Errorf("at least one image must be specified")
	}

	// This is one of the very few edge cases of baseDeploymentGenerator.
	// We accept Command and Args parameters but *only* if the number of
	// images is 1.
	if len(b.Command) > 0 || len(b.Args) > 0 {
		if len(b.Images) != 1 {
			return fmt.Errorf("command or args may only be specified if the number of images is exactly 1")
		}
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
	replicas int32,
	err error,
) {
	err = b.validate()
	if err != nil {
		return
	}
	limits, err := populateResourceListV1(b.Limits)
	if err != nil {
		return
	}
	requests, err := populateResourceListV1(b.Requests)
	if err != nil {
		return
	}
	resourceRequirements := v1.ResourceRequirements{
		Limits:   limits,
		Requests: requests,
	}
	podSpec = b.buildPodSpec(resourceRequirements)

	labels = map[string]string{"app": b.Name}
	selector = metav1.LabelSelector{MatchLabels: labels}
	replicas = b.Replicas
	return
}

func getContainerName(imageString string) string {
	// Retain just the container name
	imageSplit := strings.Split(imageString, "/")
	imageName := imageSplit[len(imageSplit)-1]
	// Remove any tag or hash
	if strings.Contains(imageName, ":") {
		imageName = strings.Split(imageName, ":")[0]
	} else if strings.Contains(imageName, "@") {
		imageName = strings.Split(imageName, "@")[0]
	}
	return imageName
}

// buildPodSpec: parse the image strings and assemble them into the Containers
// of a PodSpec. This is all you need to create the PodSpec for a deployment.
func (b BaseDeploymentGenerator) buildPodSpec(
	resourceRequirements v1.ResourceRequirements,
) v1.PodSpec {

	podSpec := v1.PodSpec{Containers: []v1.Container{}}
	for _, imageString := range b.Images {
		containerName := getContainerName(imageString)
		podSpec.Containers = append(podSpec.Containers, v1.Container{
			Name:      containerName,
			Image:     imageString,
			Resources: resourceRequirements,
			Command:   b.Command,
			Args:      b.Args,
		})
	}
	return podSpec
}

// DeploymentBasicGeneratorV1 supports stable generation of a deployment
type DeploymentBasicGeneratorV1 struct {
	BaseDeploymentGenerator
}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ StructuredGenerator = &DeploymentBasicGeneratorV1{}

// StructuredGenerate outputs a deployment object using the configured fields
func (s *DeploymentBasicGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	podSpec, labels, selector, replicas, err := s.structuredGenerate()
	return &extensionsv1beta1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   s.Name,
			Labels: labels,
		},
		Spec: extensionsv1beta1.DeploymentSpec{
			Replicas: &replicas,
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

// DeploymentBasicAppsGeneratorV1 supports stable generation of a deployment under apps/v1beta1 endpoint
type DeploymentBasicAppsGeneratorV1 struct {
	BaseDeploymentGenerator
}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ StructuredGenerator = &DeploymentBasicAppsGeneratorV1{}

// StructuredGenerate outputs a deployment object using the configured fields
func (s *DeploymentBasicAppsGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	podSpec, labels, selector, replicas, err := s.structuredGenerate()
	return &appsv1beta1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:   s.Name,
			Labels: labels,
		},
		Spec: appsv1beta1.DeploymentSpec{
			Replicas: &replicas,
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

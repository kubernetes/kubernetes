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
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestBaseDeploymentGenerator_validate(t *testing.T) {
	// Valid params should not result in an error.
	b := BaseDeploymentGenerator{
		Name:    "my-deployment",
		Images:  []string{"nginx"},
		Command: []string{"/bin/bash"},
	}
	assert.NoError(t, b.validate())

	// You should not be able to specify a Command when there are multiple
	// Images.
	b = BaseDeploymentGenerator{
		Name:    "my-deployment",
		Images:  []string{"nginx", "alpine"},
		Command: []string{"/bin/bash"},
	}
	assert.Error(t, b.validate())

	// But multiple Images with no Command is fine.
	b = BaseDeploymentGenerator{
		Name:   "my-deployment",
		Images: []string{"nginx", "alpine"},
	}
	assert.NoError(t, b.validate())
}

func TestBaseDeploymentGenerator_structuredGenerator(t *testing.T) {
	baseGenerator := BaseDeploymentGenerator{
		Name:     "hello-world",
		Images:   []string{"nginx@v1"},
		Replicas: 9,
	}

	podSpec, labels, selector, replicas, err := baseGenerator.structuredGenerate()
	assert.NoError(t, err)
	assert.Equal(t, v1.PodSpec{
		Containers: []v1.Container{{
			Name:  "nginx",
			Image: "nginx@v1",
		}},
	}, podSpec)
	expectedLabels := map[string]string{
		"app": "hello-world",
	}
	assert.Equal(t, expectedLabels, labels)
	assert.Equal(t, metav1.LabelSelector{MatchLabels: expectedLabels}, selector)
	assert.Equal(t, int32(9), replicas)
}

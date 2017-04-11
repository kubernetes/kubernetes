/*
Copyright 2015 The Kubernetes Authors.

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

package custommetrics

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api/v1"
)

func TestGetCAdvisorCustomMetricsDefinitionPath(t *testing.T) {

	regularContainer := &v1.Container{
		Name: "test_container",
	}

	cmContainer := &v1.Container{
		Name: "test_container",
		VolumeMounts: []v1.VolumeMount{
			{
				Name:      "cm",
				MountPath: CustomMetricsDefinitionDir,
			},
		},
	}
	path, err := GetCAdvisorCustomMetricsDefinitionPath(regularContainer)
	assert.Nil(t, path)
	assert.NoError(t, err)

	path, err = GetCAdvisorCustomMetricsDefinitionPath(cmContainer)
	assert.NotEmpty(t, *path)
	assert.NoError(t, err)
}

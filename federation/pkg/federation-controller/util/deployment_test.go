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

package util

import (
	"testing"

	extensionsv1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	deputils "k8s.io/kubernetes/pkg/controller/deployment/util"

	"github.com/stretchr/testify/assert"
)

func TestDeploymentEquivalent(t *testing.T) {
	d1 := newDeployment()
	d2 := newDeployment()
	d2.Annotations = make(map[string]string)

	d3 := newDeployment()
	d3.Annotations = map[string]string{"a": "b"}

	d4 := newDeployment()
	d4.Annotations = map[string]string{deputils.RevisionAnnotation: "9"}

	assert.True(t, DeploymentEquivalent(d1, d2))
	assert.True(t, DeploymentEquivalent(d1, d2))
	assert.True(t, DeploymentEquivalent(d1, d4))
	assert.True(t, DeploymentEquivalent(d4, d1))
	assert.False(t, DeploymentEquivalent(d3, d4))
	assert.False(t, DeploymentEquivalent(d3, d1))
	assert.True(t, DeploymentEquivalent(d3, d3))
}

func TestDeploymentCopy(t *testing.T) {
	d1 := newDeployment()
	d1.Annotations = map[string]string{deputils.RevisionAnnotation: "9", "a": "b"}
	d2 := DeepCopyDeployment(d1)
	assert.True(t, DeploymentEquivalent(d1, d2))
	assert.Contains(t, d2.Annotations, "a")
	assert.NotContains(t, d2.Annotations, deputils.RevisionAnnotation)
}

func newDeployment() *extensionsv1.Deployment {
	replicas := int32(5)
	return &extensionsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "wrr",
			Namespace: metav1.NamespaceDefault,
			SelfLink:  "/api/v1/namespaces/default/deployments/name123",
		},
		Spec: extensionsv1.DeploymentSpec{
			Replicas: &replicas,
		},
	}
}

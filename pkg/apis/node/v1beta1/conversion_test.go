/*
Copyright The Kubernetes Authors.

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

package v1beta1

import (
	"testing"

	"github.com/stretchr/testify/require"
	nodev1beta1 "k8s.io/api/node/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/node"
)

func TestRuntimeClassConversionClearsCheckpointPolicy(t *testing.T) {
	in := &nodev1beta1.RuntimeClass{ObjectMeta: metav1.ObjectMeta{Name: "runc"}, Handler: "runc"}
	out := &node.RuntimeClass{PodCheckpoint: &node.RuntimeClassPodCheckpoint{
		AllowedCheckpointOptions: []string{"compression"},
		AllowedRestoreOptions:    []string{"tcp-close"},
	}}
	require.NoError(t, Convert_v1beta1_RuntimeClass_To_node_RuntimeClass(in, out, nil))
	require.Nil(t, out.PodCheckpoint)
	require.Equal(t, in.Name, out.Name)
	require.Equal(t, in.Handler, out.Handler)
}

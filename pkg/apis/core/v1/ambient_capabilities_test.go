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

package v1_test

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/apis/core"
	corev1 "k8s.io/kubernetes/pkg/apis/core/v1"
)

func TestAmbientCapabilitiesSerialization(t *testing.T) {
	var original v1.Capabilities
	require.NoError(t, json.Unmarshal([]byte(`{"drop":["ALL"],"ambient":["NET_BIND_SERVICE"]}`), &original))
	require.Equal(t, []v1.Capability{"NET_BIND_SERVICE"}, original.Ambient)
	data, err := original.Marshal()
	require.NoError(t, err)
	var decoded v1.Capabilities
	require.NoError(t, decoded.Unmarshal(data))
	require.Equal(t, original, decoded)
	var internal core.Capabilities
	require.NoError(t, corev1.Convert_v1_Capabilities_To_core_Capabilities(&decoded, &internal, nil))
	require.Equal(t, []core.Capability{"NET_BIND_SERVICE"}, internal.Ambient)
	var roundTrip v1.Capabilities
	require.NoError(t, corev1.Convert_core_Capabilities_To_v1_Capabilities(&internal, &roundTrip, nil))
	require.Equal(t, original, roundTrip)
	copy := original.DeepCopy()
	copy.Ambient[0] = "CHOWN"
	require.Equal(t, v1.Capability("NET_BIND_SERVICE"), original.Ambient[0])
	internalCopy := internal.DeepCopy()
	internalCopy.Ambient[0] = "CHOWN"
	require.Equal(t, core.Capability("NET_BIND_SERVICE"), internal.Ambient[0])
}

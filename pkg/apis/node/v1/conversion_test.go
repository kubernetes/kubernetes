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
	"testing"

	"github.com/stretchr/testify/require"
	nodev1 "k8s.io/api/node/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/node"
	"k8s.io/kubernetes/pkg/apis/node/install"
)

func TestRuntimeClassCheckpointPolicyRoundTrip(t *testing.T) {
	scheme := runtime.NewScheme()
	install.Install(scheme)
	codecs := serializer.NewCodecFactory(scheme)
	in := &node.RuntimeClass{
		ObjectMeta: metav1.ObjectMeta{Name: "runc"}, Handler: "runc",
		PodCheckpoint: &node.RuntimeClassPodCheckpoint{
			AllowedCheckpointOptions: []string{"compression", "tcp-established"},
			AllowedRestoreOptions:    []string{"tcp-close"},
		},
	}
	for _, mediaType := range []string{runtime.ContentTypeJSON, runtime.ContentTypeProtobuf} {
		t.Run(mediaType, func(t *testing.T) {
			info, ok := runtime.SerializerInfoForMediaType(codecs.SupportedMediaTypes(), mediaType)
			require.True(t, ok)
			encoder := codecs.EncoderForVersion(info.Serializer, nodev1.SchemeGroupVersion)
			data, err := runtime.Encode(encoder, in)
			require.NoError(t, err)
			decoder := codecs.DecoderToVersion(info.Serializer, node.SchemeGroupVersion)
			out, err := runtime.Decode(decoder, data)
			require.NoError(t, err)
			require.Equal(t, in, out)
		})
	}
}

func TestRuntimeClassCheckpointPolicyDeepCopy(t *testing.T) {
	in := &nodev1.RuntimeClass{PodCheckpoint: &nodev1.RuntimeClassPodCheckpoint{
		AllowedCheckpointOptions: []string{"compression"},
		AllowedRestoreOptions:    []string{"tcp-close"},
	}}
	out := in.DeepCopy()
	out.PodCheckpoint.AllowedCheckpointOptions[0] = "changed-checkpoint"
	out.PodCheckpoint.AllowedRestoreOptions[0] = "changed-restore"
	require.Equal(t, "compression", in.PodCheckpoint.AllowedCheckpointOptions[0])
	require.Equal(t, "tcp-close", in.PodCheckpoint.AllowedRestoreOptions[0])
}

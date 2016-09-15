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

package secret

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"

	"github.com/stretchr/testify/assert"
)

func newTestManager(kubeClient clientset.Interface) *manager {
	return NewManager(kubeClient).(*manager)
}

func TestGetSecretsForPod(t *testing.T) {
	manager := newTestManager(&fake.Clientset{})
	secret := api.Secret{
		ObjectMeta: api.ObjectMeta{
			Namespace: "test",
			Name:      "aa",
		},
		Data: map[string][]byte{
			"data-1": []byte("value-1"),
			"data-2": []byte("value-2"),
		},
	}
	manager.secretStore.Add(&secret)
	pod := &api.Pod{
		ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      "foo",
			Namespace: "test",
		},
		Spec: api.PodSpec{
			ImagePullSecrets: []api.LocalObjectReference{
				api.LocalObjectReference{
					Name: "aa",
				},
			},
		},
	}
	secrets, _ := manager.GetPullSecretsForPod(pod)
	assert.Equal(t, len(secrets), 1, "Unexpeted %+v", secrets)
}

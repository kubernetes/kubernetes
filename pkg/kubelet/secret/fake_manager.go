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
	"fmt"

	v1 "k8s.io/api/core/v1"
)

// fakeManager implements Manager interface for testing purposes.
// simple operations to apiserver.
type fakeManager struct {
	secrets []*v1.Secret
}

// NewFakeManager creates empty/fake secret manager
func NewFakeManager() Manager {
	return &fakeManager{}
}

// NewFakeManagerWithSecrets creates a fake secret manager with the provided secrets
func NewFakeManagerWithSecrets(secrets []*v1.Secret) Manager {
	return &fakeManager{
		secrets: secrets,
	}
}

// GetSecret function returns the searched secret if it was provided during the manager initialization, otherwise, it returns an error.
// If the manager was initialized without any secrets, it returns a nil secret."
func (s *fakeManager) GetSecret(namespace, name string) (*v1.Secret, error) {
	if s.secrets == nil {
		return nil, nil
	}

	for _, secret := range s.secrets {
		if secret.Name == name {
			return secret, nil
		}
	}

	return nil, fmt.Errorf("secret %s not found", name)
}

// RegisterPod implements the RegisterPod method for testing purposes.
func (s *fakeManager) RegisterPod(pod *v1.Pod) {
}

// UnregisterPod implements the UnregisterPod method for testing purposes.
func (s *fakeManager) UnregisterPod(pod *v1.Pod) {
}

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

package resourcelock

import (
	"encoding/json"
	"errors"
	"fmt"

	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	clientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
)

type SecretsLock struct {
	// SecretsMeta should contain a Name and a Namespace of a
	// Secret object that the LeaderElector will attempt to lead.
	SecretsMeta v1.ObjectMeta
	Client        clientset.Interface
	LockConfig    ResourceLockConfig
	e             *v1.Secret
}

func (el *SecretsLock) Get() (*LeaderElectionRecord, error) {
	var record LeaderElectionRecord
	var err error
	el.e, err = el.Client.Core().Secrets(el.SecretsMeta.Namespace).Get(el.SecretsMeta.Name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	if el.e.Annotations == nil {
		el.e.Annotations = make(map[string]string)
	}
	if recordBytes, found := el.e.Annotations[LeaderElectionRecordAnnotationKey]; found {
		if err := json.Unmarshal([]byte(recordBytes), &record); err != nil {
			return nil, err
		}
	}
	return &record, nil
}

// Create attempts to create a LeaderElectionRecord annotation
func (el *SecretsLock) Create(ler LeaderElectionRecord) error {
	recordBytes, err := json.Marshal(ler)
	if err != nil {
		return err
	}
	el.e, err = el.Client.Core().Secrets(el.SecretsMeta.Namespace).Create(&v1.Secret{
		ObjectMeta: v1.ObjectMeta{
			Name:      el.SecretsMeta.Name,
			Namespace: el.SecretsMeta.Namespace,
			Annotations: map[string]string{
				LeaderElectionRecordAnnotationKey: string(recordBytes),
			},
		},
	})
	return err
}

// Update will update and existing annotation on a given resource.
func (el *SecretsLock) Update(ler LeaderElectionRecord) error {
	if el.e == nil {
		return errors.New("secret not initialized, call get or create first")
	}
	recordBytes, err := json.Marshal(ler)
	if err != nil {
		return err
	}
	el.e.Annotations[LeaderElectionRecordAnnotationKey] = string(recordBytes)
	el.e, err = el.Client.Core().Secrets(el.SecretsMeta.Namespace).Update(el.e)
	return err
}

// Describe is used to convert details on current resource lock
// into a string
func (el *SecretsLock) Describe() string {
	return fmt.Sprintf("%v/%v", el.SecretsMeta.Namespace, el.SecretsMeta.Name)
}

// returns the Identity of the lock
func (el *SecretsLock) Identity() string {
	return el.LockConfig.Identity
}

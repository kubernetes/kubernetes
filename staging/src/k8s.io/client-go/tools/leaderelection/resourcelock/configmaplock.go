/*
Copyright 2017 The Kubernetes Authors.

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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
)

// TODO: This is almost a exact replica of Endpoints lock.
// going forwards as we self host more and more components
// and use ConfigMaps as the means to pass that configuration
// data we will likely move to deprecate the Endpoints lock.

type ConfigMapLock struct {
	// ConfigMapMeta should contain a Name and a Namespace of a
	// ConfigMapMeta object that the LeaderElector will attempt to lead.
	ConfigMapMeta metav1.ObjectMeta
	Client        corev1client.ConfigMapsGetter
	LockConfig    ResourceLockConfig
	cm            *v1.ConfigMap
}

// Get returns the election record from a ConfigMap Annotation
func (cml *ConfigMapLock) Get() (*LeaderElectionRecord, error) {
	var record LeaderElectionRecord
	var err error
	cml.cm, err = cml.Client.ConfigMaps(cml.ConfigMapMeta.Namespace).Get(cml.ConfigMapMeta.Name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	if cml.cm.Annotations == nil {
		cml.cm.Annotations = make(map[string]string)
	}
	if recordBytes, found := cml.cm.Annotations[LeaderElectionRecordAnnotationKey]; found {
		if err := json.Unmarshal([]byte(recordBytes), &record); err != nil {
			return nil, err
		}
	}
	return &record, nil
}

// Create attempts to create a LeaderElectionRecord annotation
func (cml *ConfigMapLock) Create(ler LeaderElectionRecord) error {
	recordBytes, err := json.Marshal(ler)
	if err != nil {
		return err
	}
	cml.cm, err = cml.Client.ConfigMaps(cml.ConfigMapMeta.Namespace).Create(&v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      cml.ConfigMapMeta.Name,
			Namespace: cml.ConfigMapMeta.Namespace,
			Annotations: map[string]string{
				LeaderElectionRecordAnnotationKey: string(recordBytes),
			},
		},
	})
	return err
}

// Update will update an existing annotation on a given resource.
func (cml *ConfigMapLock) Update(ler LeaderElectionRecord) error {
	if cml.cm == nil {
		return errors.New("configmap not initialized, call get or create first")
	}
	recordBytes, err := json.Marshal(ler)
	if err != nil {
		return err
	}
	cml.cm.Annotations[LeaderElectionRecordAnnotationKey] = string(recordBytes)
	cml.cm, err = cml.Client.ConfigMaps(cml.ConfigMapMeta.Namespace).Update(cml.cm)
	return err
}

// RecordEvent in leader election while adding meta-data
func (cml *ConfigMapLock) RecordEvent(s string) {
	events := fmt.Sprintf("%v %v", cml.LockConfig.Identity, s)
	cml.LockConfig.EventRecorder.Eventf(&v1.ConfigMap{ObjectMeta: cml.cm.ObjectMeta}, v1.EventTypeNormal, "LeaderElection", events)
}

// Describe is used to convert details on current resource lock
// into a string
func (cml *ConfigMapLock) Describe() string {
	return fmt.Sprintf("%v/%v", cml.ConfigMapMeta.Namespace, cml.ConfigMapMeta.Name)
}

// returns the Identity of the lock
func (cml *ConfigMapLock) Identity() string {
	return cml.LockConfig.Identity
}

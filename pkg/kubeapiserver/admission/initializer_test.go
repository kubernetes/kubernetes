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

package admission

import (
	"context"
	"testing"

	"k8s.io/apiserver/pkg/admission"
)

type doNothingAdmission struct{}

func (doNothingAdmission) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	return nil
}
func (doNothingAdmission) Handles(o admission.Operation) bool { return false }
func (doNothingAdmission) Validate() error                    { return nil }

type WantsCloudConfigAdmissionPlugin struct {
	doNothingAdmission
	cloudConfig []byte
}

func (p *WantsCloudConfigAdmissionPlugin) SetCloudConfig(cloudConfig []byte) {
	p.cloudConfig = cloudConfig
}

func TestCloudConfigAdmissionPlugin(t *testing.T) {
	cloudConfig := []byte("cloud-configuration")
	initializer := NewPluginInitializer(cloudConfig, nil, nil)
	wantsCloudConfigAdmission := &WantsCloudConfigAdmissionPlugin{}
	initializer.Initialize(wantsCloudConfigAdmission)

	if wantsCloudConfigAdmission.cloudConfig == nil {
		t.Errorf("Expected cloud config to be initialized but found nil")
	}
}

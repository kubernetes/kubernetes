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
	"net/url"
	"testing"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
)

type doNothingAdmission struct{}

func (doNothingAdmission) Admit(a admission.Attributes) error { return nil }
func (doNothingAdmission) Handles(o admission.Operation) bool { return false }
func (doNothingAdmission) Validate() error                    { return nil }

type WantsCloudConfigAdmissionPlugin struct {
	doNothingAdmission
	cloudConfig []byte
}

func (self *WantsCloudConfigAdmissionPlugin) SetCloudConfig(cloudConfig []byte) {
	self.cloudConfig = cloudConfig
}

func TestCloudConfigAdmissionPlugin(t *testing.T) {
	cloudConfig := []byte("cloud-configuration")
	initializer := NewPluginInitializer(nil, nil, cloudConfig, nil, nil, nil, nil)
	wantsCloudConfigAdmission := &WantsCloudConfigAdmissionPlugin{}
	initializer.Initialize(wantsCloudConfigAdmission)

	if wantsCloudConfigAdmission.cloudConfig == nil {
		t.Errorf("Expected cloud config to be initialized but found nil")
	}
}

type fakeServiceResolver struct{}

func (f *fakeServiceResolver) ResolveEndpoint(namespace, name string) (*url.URL, error) {
	return nil, nil
}

type serviceWanter struct {
	doNothingAdmission
	got ServiceResolver
}

func (s *serviceWanter) SetServiceResolver(sr webhook.ServiceResolver) { s.got = sr }

func TestWantsServiceResolver(t *testing.T) {
	sw := &serviceWanter{}
	fsr := &fakeServiceResolver{}
	i := NewPluginInitializer(nil, nil, nil, nil, nil, nil, fsr)
	i.Initialize(sw)
	if got, ok := sw.got.(*fakeServiceResolver); !ok || got != fsr {
		t.Errorf("plumbing fail - %v %v#", ok, got)
	}
}

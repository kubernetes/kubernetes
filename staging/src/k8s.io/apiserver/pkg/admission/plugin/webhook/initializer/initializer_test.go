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

package initializer

import (
	"net/url"
	"testing"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config"
)

type doNothingAdmission struct{}

func (doNothingAdmission) Admit(a admission.Attributes) error { return nil }
func (doNothingAdmission) Handles(o admission.Operation) bool { return false }
func (doNothingAdmission) Validate() error                    { return nil }

type fakeServiceResolver struct{}

func (f *fakeServiceResolver) ResolveEndpoint(namespace, name string) (*url.URL, error) {
	return nil, nil
}

type serviceWanter struct {
	doNothingAdmission
	got ServiceResolver
}

func (s *serviceWanter) SetServiceResolver(sr config.ServiceResolver) { s.got = sr }

func TestWantsServiceResolver(t *testing.T) {
	sw := &serviceWanter{}
	fsr := &fakeServiceResolver{}
	i := NewPluginInitializer(nil, fsr)
	i.Initialize(sw)
	if got, ok := sw.got.(*fakeServiceResolver); !ok || got != fsr {
		t.Errorf("plumbing fail - %v %v#", ok, got)
	}
}

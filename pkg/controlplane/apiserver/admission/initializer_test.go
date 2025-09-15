/*
Copyright 2024 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	quota "k8s.io/apiserver/pkg/quota/v1"
)

type doNothingAdmission struct{}

func (doNothingAdmission) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	return nil
}
func (doNothingAdmission) Handles(o admission.Operation) bool { return false }
func (doNothingAdmission) Validate() error                    { return nil }

type doNothingPluginInitialization struct{}

func (doNothingPluginInitialization) ValidateInitialization() error { return nil }

type doNothingQuotaConfiguration struct{}

func (doNothingQuotaConfiguration) IgnoredResources() map[schema.GroupResource]struct{} { return nil }

func (doNothingQuotaConfiguration) Evaluators() []quota.Evaluator { return nil }

type WantsQuotaConfigurationAdmissionPlugin struct {
	doNothingAdmission
	doNothingPluginInitialization
	config quota.Configuration
}

func (p *WantsQuotaConfigurationAdmissionPlugin) SetQuotaConfiguration(config quota.Configuration) {
	p.config = config
}

func TestQuotaConfigurationAdmissionPlugin(t *testing.T) {
	config := doNothingQuotaConfiguration{}
	initializer := NewPluginInitializer(config, nil)
	wantsQuotaConfigurationAdmission := &WantsQuotaConfigurationAdmissionPlugin{}
	initializer.Initialize(wantsQuotaConfigurationAdmission)

	if wantsQuotaConfigurationAdmission.config == nil {
		t.Errorf("Expected quota configuration to be initialized but found nil")
	}
}

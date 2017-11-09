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

package validation

import (
	internalapi "k8s.io/kubernetes/plugin/pkg/admission/initialresources/apis/initialresources"
	"testing"
)

func TestValidateConfiguration(t *testing.T) {

	testCases := []struct {
		TestName  string
		Config    internalapi.Configuration
		ExpectErr bool
	}{
		{
			TestName: "valid case",
			Config: internalapi.Configuration{
				DataSourceInfo: internalapi.DataSourceInfo{
					DataSource: internalapi.Influxdb,
				},
			},
			ExpectErr: false,
		},
		{
			TestName: "invalid case",
			Config: internalapi.Configuration{
				DataSourceInfo: internalapi.DataSourceInfo{
					DataSource: internalapi.DataSourceType("invalid"),
				},
			},
			ExpectErr: true,
		},
	}

	for i := range testCases {
		errs := ValidateConfiguration(&testCases[i].Config)
		if !testCases[i].ExpectErr && errs != nil {
			t.Errorf("Test: %s, expected success: %v", testCases[i].TestName, errs)
		}
		if testCases[i].ExpectErr && errs == nil {
			t.Errorf("Test: %s, expected errors: %v", testCases[i].TestName, errs)
		}
	}
}

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

package defaults

import (
	"os"
	"testing"
)

func TestGetMaxVols(t *testing.T) {
	os.Setenv("KUBE_MAX_PD_VOLS", "")
	defaultValue := 39
	result := getMaxVols(defaultValue)
	if result != defaultValue {
		t.Errorf("%s: expected %v got %v", "Unable to parse maxiumum PD volumes value,using default value ", defaultValue, result)
	}

	os.Setenv("KUBE_MAX_PD_VOLS", "-2")
	result = getMaxVols(defaultValue)
	if result != defaultValue {
		t.Errorf("%s: expected %v got %v", "Maximum PD volumes must be a positive value, using default ", defaultValue, result)
	}

	os.Setenv("KUBE_MAX_PD_VOLS", "40")
	result = getMaxVols(defaultValue)
	if result != defaultValue {
		t.Errorf("%s: expected %v got %v", "Parse maximum PD volumes value from env", defaultValue, result)
	}

	os.Unsetenv("KUBE_MAX_PD_VOLS")

}

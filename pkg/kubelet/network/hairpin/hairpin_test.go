/*
Copyright 2015 The Kubernetes Authors.

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

package hairpin

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

func TestSetUpInterfaceNonExistent(t *testing.T) {
	err := setUpInterface("non-existent")
	if err == nil {
		t.Errorf("unexpected non-error")
	}
	deviceDir := fmt.Sprintf("%s/%s", sysfsNetPath, "non-existent")
	if !strings.Contains(fmt.Sprintf("%v", err), deviceDir) {
		t.Errorf("should have tried to open %s", deviceDir)
	}
}

func TestSetUpInterfaceNotBridged(t *testing.T) {
	err := setUpInterface("lo")
	if err != nil {
		if os.IsNotExist(err) {
			t.Skipf("'lo' device does not exist??? (%v)", err)
		}
		t.Errorf("unexpected error: %v", err)
	}
}

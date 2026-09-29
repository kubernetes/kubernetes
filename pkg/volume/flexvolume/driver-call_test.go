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

package flexvolume

import (
	"errors"
	"testing"
	"time"

	"k8s.io/utils/exec"
	exectesting "k8s.io/utils/exec/testing"
)

func TestHandleResponseDefaults(t *testing.T) {
	ds, err := handleCmdResponse("test", []byte(`{"status": "Success"}`))
	if err != nil {
		t.Error("error: ", err)
	}

	if *ds.Capabilities != *defaultCapabilities() {
		t.Error("wrong default capabilities: ", *ds.Capabilities)
	}
}

func TestDriverCallTimeout(t *testing.T) {
	plugin := &flexVolumePlugin{
		driverName: "test",
		execPath:   "/plugin",
		runner: fakeRunner(func(_ string, _ ...string) exec.Cmd {
			return &exectesting.FakeCmd{
				CombinedOutputScript: []exectesting.FakeAction{
					func() ([]byte, []byte, error) {
						// Let the timeout callback run without synchronizing with its flag update.
						time.Sleep(20 * time.Millisecond)
						return nil, nil, errors.New("command stopped")
					},
				},
			}
		}),
	}

	call := plugin.NewDriverCallWithTimeout("test", time.Millisecond)
	_, err := call.Run()
	if err != errTimeout {
		t.Fatalf("expected timeout error %v, got %v", errTimeout, err)
	}
}

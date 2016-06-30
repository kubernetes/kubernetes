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

package container

import (
	"errors"
	"testing"
)

func TestPodSyncResult(t *testing.T) {
	okResults := []*SyncResult{
		NewSyncResult(StartContainer, "container_0"),
		NewSyncResult(SetupNetwork, "pod"),
	}
	errResults := []*SyncResult{
		NewSyncResult(KillContainer, "container_1"),
		NewSyncResult(TeardownNetwork, "pod"),
	}
	errResults[0].Fail(errors.New("error_0"), "message_0")
	errResults[1].Fail(errors.New("error_1"), "message_1")

	// If the PodSyncResult doesn't contain error result, it should not be error
	result := PodSyncResult{}
	result.AddSyncResult(okResults...)
	if result.Error() != nil {
		t.Errorf("PodSyncResult should not be error: %v", result)
	}

	// If the PodSyncResult contains error result, it should be error
	result = PodSyncResult{}
	result.AddSyncResult(okResults...)
	result.AddSyncResult(errResults...)
	if result.Error() == nil {
		t.Errorf("PodSyncResult should be error: %q", result)
	}

	// If the PodSyncResult is failed, it should be error
	result = PodSyncResult{}
	result.AddSyncResult(okResults...)
	result.Fail(errors.New("error"))
	if result.Error() == nil {
		t.Errorf("PodSyncResult should be error: %q", result)
	}

	// If the PodSyncResult is added an error PodSyncResult, it should be error
	errResult := PodSyncResult{}
	errResult.AddSyncResult(errResults...)
	result = PodSyncResult{}
	result.AddSyncResult(okResults...)
	result.AddPodSyncResult(errResult)
	if result.Error() == nil {
		t.Errorf("PodSyncResult should be error: %q", result)
	}
}

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

package oom

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
)

var oomWatcherRequiredFiles = []string{
	"/dev/kmsg",
}

func TestBasicIfPossible(t *testing.T) {
	if requiredAccessibleFilesToRunTest() {
		fakeRecorder := &record.FakeRecorder{}
		node := &v1.ObjectReference{}
		oomWatcher := NewOOMWatcher(fakeRecorder)
		assert.NoError(t, oomWatcher.Start(node))
		// TODO: Improve this test once cadvisor exports events.EventChannel as an interface
		// and thereby allow using a mock version of cadvisor.
	} else {
		t.Logf("Skipping test as required files don't exist or we don't have access to them.")
		assert.True(t, true)
	}
}

func requiredAccessibleFilesToRunTest() bool {
	passesFileCheck := true

	for _, fileName := range oomWatcherRequiredFiles {
		if _, err := os.Stat(fileName); os.IsNotExist(err) {
			passesFileCheck = false
		}

		file, err := os.OpenFile(fileName, os.O_RDONLY, 0666)
		if err != nil {
			if os.IsPermission(err) {
				passesFileCheck = false
			}
		}
		file.Close()
	}

	return passesFileCheck
}

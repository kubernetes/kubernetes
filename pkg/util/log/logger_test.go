/*
Copyright 2019 The Kubernetes Authors.

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

package log

import (
	"fmt"
	"testing"
	"time"
)

func TestLogger(t *testing.T) {
	msg := "foo"
	expectedMsg := msg
	actualMsg := getMessageInfrequently(msg)
	if expectedMsg != actualMsg {
		t.Errorf("First message is altered. Expected: %s, got: %s", expectedMsg, actualMsg)
	}

	expectedMsg = ""
	actualMsg = getMessageInfrequently(msg)
	if expectedMsg != actualMsg {
		t.Errorf("Second message not empty. Got: %s", actualMsg)
	}

	expectedMsg = fmt.Sprintf("(x3) %s", msg)
	entry, _ := messages[msg]
	entry.timestamp = time.Now().Add(-10 * time.Second)
	actualMsg = getMessageInfrequently(msg)
	if expectedMsg != actualMsg {
		t.Errorf("Third logged message not as expected. Expected: %s, got: %s", expectedMsg, actualMsg)
	}

	expectedMsg = ""
	actualMsg = getMessageInfrequently(msg)
	if expectedMsg != actualMsg {
		t.Errorf("Fourth message not empty. Got: %s", actualMsg)
	}

	expectedMsg = fmt.Sprintf("(x5) %s", msg)
	entry.timestamp = time.Now().Add(-300 * time.Second)
	entry.countLogged = 100
	actualMsg = getMessageInfrequently(msg)
	if expectedMsg != actualMsg {
		t.Errorf("Fifth logged message not as expected. Expected: %s, got: %s", expectedMsg, actualMsg)
	}

	expectedMsg = msg
	entry.timestamp = time.Now().Add(-10 * time.Minute)
	cleanOldMessages()
	actualMsg = getMessageInfrequently(msg)
	if expectedMsg != actualMsg {
		t.Errorf("First message after cleaunup is altered. Expected: %s, got: %s", expectedMsg, actualMsg)
	}
}

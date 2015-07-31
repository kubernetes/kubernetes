/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// Channel Manager keeps track of multiple channels
package operationmanager

import (
	"testing"
)

func TestStart(t *testing.T) {
	// Arrange
	cm := NewOperationManager()
	chanId := "testChanId"
	testMsg := "test message"

	// Act
	ch, startErr := cm.Start(chanId, 1 /* bufferSize */)
	sigErr := cm.Send(chanId, testMsg)

	// Assert
	verifyNoError(t, startErr, "Start")
	verifyNoError(t, sigErr, "Send")
	actualMsg := <-ch
	verifyMsg(t, testMsg /* expected */, actualMsg.(string) /* actual */)
}

func TestStartIdExists(t *testing.T) {
	// Arrange
	cm := NewOperationManager()
	chanId := "testChanId"

	// Act
	_, startErr1 := cm.Start(chanId, 1 /* bufferSize */)
	_, startErr2 := cm.Start(chanId, 1 /* bufferSize */)

	// Assert
	verifyNoError(t, startErr1, "Start1")
	verifyError(t, startErr2, "Start2")
}

func TestStartAndAdd2Chans(t *testing.T) {
	// Arrange
	cm := NewOperationManager()
	chanId1 := "testChanId1"
	chanId2 := "testChanId2"
	testMsg1 := "test message 1"
	testMsg2 := "test message 2"

	// Act
	ch1, startErr1 := cm.Start(chanId1, 1 /* bufferSize */)
	ch2, startErr2 := cm.Start(chanId2, 1 /* bufferSize */)
	sigErr1 := cm.Send(chanId1, testMsg1)
	sigErr2 := cm.Send(chanId2, testMsg2)

	// Assert
	verifyNoError(t, startErr1, "Start1")
	verifyNoError(t, startErr2, "Start2")
	verifyNoError(t, sigErr1, "Send1")
	verifyNoError(t, sigErr2, "Send2")
	actualMsg1 := <-ch1
	actualMsg2 := <-ch2
	verifyMsg(t, testMsg1 /* expected */, actualMsg1.(string) /* actual */)
	verifyMsg(t, testMsg2 /* expected */, actualMsg2.(string) /* actual */)
}

func TestStartAndAdd2ChansAndClose(t *testing.T) {
	// Arrange
	cm := NewOperationManager()
	chanId1 := "testChanId1"
	chanId2 := "testChanId2"
	testMsg1 := "test message 1"
	testMsg2 := "test message 2"

	// Act
	ch1, startErr1 := cm.Start(chanId1, 1 /* bufferSize */)
	ch2, startErr2 := cm.Start(chanId2, 1 /* bufferSize */)
	sigErr1 := cm.Send(chanId1, testMsg1)
	sigErr2 := cm.Send(chanId2, testMsg2)
	cm.Close(chanId1)
	sigErr3 := cm.Send(chanId1, testMsg1)

	// Assert
	verifyNoError(t, startErr1, "Start1")
	verifyNoError(t, startErr2, "Start2")
	verifyNoError(t, sigErr1, "Send1")
	verifyNoError(t, sigErr2, "Send2")
	verifyError(t, sigErr3, "Send3")
	actualMsg1 := <-ch1
	actualMsg2 := <-ch2
	verifyMsg(t, testMsg1 /* expected */, actualMsg1.(string) /* actual */)
	verifyMsg(t, testMsg2 /* expected */, actualMsg2.(string) /* actual */)
}

func TestExists(t *testing.T) {
	// Arrange
	cm := NewOperationManager()
	chanId1 := "testChanId1"
	chanId2 := "testChanId2"

	// Act & Assert
	verifyExists(t, cm, chanId1, false /* expected */)
	verifyExists(t, cm, chanId2, false /* expected */)

	_, startErr1 := cm.Start(chanId1, 1 /* bufferSize */)
	verifyNoError(t, startErr1, "Start1")
	verifyExists(t, cm, chanId1, true /* expected */)
	verifyExists(t, cm, chanId2, false /* expected */)

	_, startErr2 := cm.Start(chanId2, 1 /* bufferSize */)
	verifyNoError(t, startErr2, "Start2")
	verifyExists(t, cm, chanId1, true /* expected */)
	verifyExists(t, cm, chanId2, true /* expected */)

	cm.Close(chanId1)
	verifyExists(t, cm, chanId1, false /* expected */)
	verifyExists(t, cm, chanId2, true /* expected */)

	cm.Close(chanId2)
	verifyExists(t, cm, chanId1, false /* expected */)
	verifyExists(t, cm, chanId2, false /* expected */)
}

func verifyExists(t *testing.T, cm OperationManager, id string, expected bool) {
	if actual := cm.Exists(id); expected != actual {
		t.Fatalf("Unexpected Exists(%q) response. Expected: <%v> Actual: <%v>", id, expected, actual)
	}
}

func verifyNoError(t *testing.T, err error, name string) {
	if err != nil {
		t.Fatalf("Unexpected response on %q. Expected: <no error> Actual: <%v>", name, err)
	}
}

func verifyError(t *testing.T, err error, name string) {
	if err == nil {
		t.Fatalf("Unexpected response on %q. Expected: <error> Actual: <no error>")
	}
}

func verifyMsg(t *testing.T, expected, actual string) {
	if actual != expected {
		t.Fatalf("Unexpected testMsg value. Expected: <%v> Actual: <%v>", expected, actual)
	}
}

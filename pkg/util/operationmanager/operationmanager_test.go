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
	if startErr != nil {
		t.Fatalf("Unexpected error on Start. Expected: <no error> Actual: <%v>", startErr)
	}
	if sigErr != nil {
		t.Fatalf("Unexpected error on Send. Expected: <no error> Actual: <%v>", sigErr)
	}
	if actual := <-ch; actual != testMsg {
		t.Fatalf("Unexpected testMsg value. Expected: <%v> Actual: <%v>", testMsg, actual)
	}
}

func TestStartIdExists(t *testing.T) {
	// Arrange
	cm := NewOperationManager()
	chanId := "testChanId"

	// Act
	_, startErr1 := cm.Start(chanId, 1 /* bufferSize */)
	_, startErr2 := cm.Start(chanId, 1 /* bufferSize */)

	// Assert
	if startErr1 != nil {
		t.Fatalf("Unexpected error on Start1. Expected: <no error> Actual: <%v>", startErr1)
	}
	if startErr2 == nil {
		t.Fatalf("Expected error on Start2. Expected: <id already exists error> Actual: <no error>")
	}
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
	if startErr1 != nil {
		t.Fatalf("Unexpected error on Start1. Expected: <no error> Actual: <%v>", startErr1)
	}
	if startErr2 != nil {
		t.Fatalf("Unexpected error on Start2. Expected: <no error> Actual: <%v>", startErr2)
	}
	if sigErr1 != nil {
		t.Fatalf("Unexpected error on Send1. Expected: <no error> Actual: <%v>", sigErr1)
	}
	if sigErr2 != nil {
		t.Fatalf("Unexpected error on Send2. Expected: <no error> Actual: <%v>", sigErr2)
	}
	if actual := <-ch1; actual != testMsg1 {
		t.Fatalf("Unexpected testMsg value. Expected: <%v> Actual: <%v>", testMsg1, actual)
	}
	if actual := <-ch2; actual != testMsg2 {
		t.Fatalf("Unexpected testMsg value. Expected: <%v> Actual: <%v>", testMsg2, actual)
	}

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
	if startErr1 != nil {
		t.Fatalf("Unexpected error on Start1. Expected: <no error> Actual: <%v>", startErr1)
	}
	if startErr2 != nil {
		t.Fatalf("Unexpected error on Start2. Expected: <no error> Actual: <%v>", startErr2)
	}
	if sigErr1 != nil {
		t.Fatalf("Unexpected error on Send1. Expected: <no error> Actual: <%v>", sigErr1)
	}
	if sigErr2 != nil {
		t.Fatalf("Unexpected error on Send2. Expected: <no error> Actual: <%v>", sigErr2)
	}
	if sigErr3 == nil {
		t.Fatalf("Expected error on Send3. Expected: <error> Actual: <no error>", sigErr2)
	}
	if actual := <-ch1; actual != testMsg1 {
		t.Fatalf("Unexpected testMsg value. Expected: <%v> Actual: <%v>", testMsg1, actual)
	}
	if actual := <-ch2; actual != testMsg2 {
		t.Fatalf("Unexpected testMsg value. Expected: <%v> Actual: <%v>", testMsg2, actual)
	}

}

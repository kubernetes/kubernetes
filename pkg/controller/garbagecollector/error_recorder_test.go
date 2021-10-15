/*
Copyright 2021 The Kubernetes Authors.

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

package garbagecollector

import (
	goerrors "errors"
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var (
	newXNode = func() *node {
		return &node{
			identity: objectReference{
				OwnerReference: metav1.OwnerReference{
					Name:       "x",
					UID:        "xnode",
					APIVersion: "xnodeapiversion",
					Kind:       "XNode",
				},
				Namespace: "xnodens",
			},
		}
	}
	newYNode = func() *node {
		return &node{
			identity: objectReference{
				OwnerReference: metav1.OwnerReference{
					Name: "y",
				},
			},
		}
	}
	newZNode = func() *node {
		return &node{
			identity: objectReference{
				OwnerReference: metav1.OwnerReference{
					Name: "z",
				},
			},
		}
	}
)

func TestErrorRecorder(t *testing.T) {
	recorder := newErrorRecorder()

	xNode, yNode, zNode := newXNode(), newYNode(), newZNode()

	recorder.SetSyncError(goerrors.New("sync failed"))
	recorder.SetAttemptToOrphanError(xNode, goerrors.New(testErrMessage("orphan", "x")))
	recorder.SetAttemptToOrphanError(yNode, goerrors.New(testErrMessage("orphan", "y")))
	recorder.ClearAttemptToOrphanError(xNode)
	recorder.SetAttemptToOrphanError(zNode, goerrors.New(testErrMessage("orphan", "z")))
	recorder.SetAttemptToDeleteError(xNode, goerrors.New(testErrMessage("delete", "x")))

	errors := recorder.DumpErrors()

	if len(errors) != 4 {
		t.Errorf("unexpected number of errors number of errors, expected %v, got %v", 4, errors)
	}

	for _, errResult := range errors {
		if errResult.ErrorTime.IsZero() {
			t.Errorf("error time is unitialized for: %v", errResult)
		}
	}

	// check sync error
	syncError := errors[0]
	if syncError.ErrorType != SyncError || syncError.Error != "sync failed" {
		t.Errorf("syncError error result not found: %v", errors)
	}

	// check delete error
	deleteError := errors[1]
	if deleteError.ErrorType != AttemptToDeleteError || deleteError.Error != testErrMessage("delete", "x") {
		t.Errorf("attemptToDelete error result not found: %v", errors)
	}
	if deleteError.InvolvedObject == nil ||
		deleteError.InvolvedObject.Name != "x" ||
		deleteError.InvolvedObject.UID != "xnode" ||
		deleteError.InvolvedObject.APIVersion != "xnodeapiversion" ||
		deleteError.InvolvedObject.Kind != "XNode" ||
		deleteError.InvolvedObject.Namespace != "xnodens" {
		t.Errorf("attemptToDelete error result InvolvedObject is invalid: %v", deleteError)
	}

	// check orphan errors
	for _, orphanError := range errors[2:] {
		isOrphanError := orphanError.ErrorType == AttemptToOrphanError && orphanError.InvolvedObject != nil && (orphanError.InvolvedObject.Name == "y" || orphanError.InvolvedObject.Name == "z")
		if !isOrphanError || orphanError.Error != testErrMessage("orphan", orphanError.InvolvedObject.Name) {
			t.Errorf("attemptToOrphan error result not found: %v", errors)
		}
	}

	recorder.ClearSyncError()
	for _, n := range []*node{xNode, yNode, zNode} {
		recorder.ClearAttemptToOrphanError(n)
		recorder.ClearAttemptToDeleteError(n)
	}

	errors = recorder.DumpErrors()

	if len(errors) != 0 {
		t.Errorf("unsuccessful clear: got unexpected %v", errors)
	}
}

func testErrMessage(action, what string) string {
	return fmt.Sprintf("attempt to %s %s failed", action, what)
}

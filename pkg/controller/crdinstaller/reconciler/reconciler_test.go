/*
Copyright 2018 The Kubernetes Authors.

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

package reconciler

import (
	fakeapiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset/fake"
	"k8s.io/kubernetes/pkg/controller/crdinstaller/crdgenerator"
	"k8s.io/kubernetes/pkg/controller/crdinstaller/crds/attachdetachcrd"
	"testing"
	"time"
	"k8s.io/apimachinery/pkg/util/wait"
)

const (
	reconcilerLoopPeriod time.Duration = 0 * time.Millisecond
)

// Calls Run()
// Verifies there are no calls to create a CRD.
func Test_Run_Positive_DoNothing(t *testing.T) {
	// Arrange
	fakeApiExtensionsClient := fakeapiextensionsclient.NewSimpleClientset()
	reconciler := NewReconciler(
		reconcilerLoopPeriod,
		fakeApiExtensionsClient, /* crdClient */
		[]crdgenerator.ControllerCRDGenerator{crdgenerator.NewFakeControllerCRDGenerator()},
	)

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Assert
	waitForCount(t, fakeApiExtensionsClient)
}

// Calls Run()
// Verifies Attach/Detach Controller CRDs create calls
func Test_Run_Positive_AttachDetachCRDs(t *testing.T) {
	// Arrange
	fakeApiExtensionsClient := fakeapiextensionsclient.NewSimpleClientset()
	reconciler := NewReconciler(
		reconcilerLoopPeriod,
		fakeApiExtensionsClient, /* crdClient */
		[]crdgenerator.ControllerCRDGenerator{attachdetachcrd.NewAttachDetachControllerCRDGenerator()},
	)

	// Act
	ch := make(chan struct{})
	go reconciler.Run(ch)
	defer close(ch)

	// Assert
}

func waitForCount(
	t *testing.T,
	fakeClient *fakeapiextensionsclient.Clientset) {
	err := retryWithExponentialBackOff(
		time.Duration(5*time.Millisecond),
		func() (bool, error) {
			// actualCallCount := fakePlugin.GetNewDetacherCallCount()
			// if actualCallCount >= expectedCallCount {
			// 	return true, nil
			// }
			t.Logf(
				"Warning: BLAH BLAH. Expected: <%#v> Actual: <%v>. Will retry.",
				fakeClient.Fake,
				nil)
			return false, nil
		},
	)

	if err != nil {
		t.Fatalf(
			"Timed out waiting for NewDetacherCallCount. Expected: <%v> Actual: <%v>", 0, 0)
			// expectedCallCount,
			//fakePlugin.GetNewDetacherCallCount())
	}
}

func retryWithExponentialBackOff(initialDuration time.Duration, fn wait.ConditionFunc) error {
	backoff := wait.Backoff{
		Duration: initialDuration,
		Factor:   3,
		Jitter:   0,
		Steps:    6,
	}
	return wait.ExponentialBackoff(backoff, fn)
}
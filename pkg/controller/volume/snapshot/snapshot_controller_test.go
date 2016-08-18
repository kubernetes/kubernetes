/*
Copyright 2016 The Kubernetes Authors.

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

package snapshot

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/controller/framework/informers"
	controllervolumetesting "k8s.io/kubernetes/pkg/controller/volume/snapshot/testing"
)

func Test_NewSnapshotController_Positive(t *testing.T) {
	// Arrange
	fakeKubeClient := controllervolumetesting.CreateTestClient()
	resyncPeriod := 5 * time.Minute
	pvcInformer := informers.NewPVCInformer(fakeKubeClient, resyncPeriod)
	pvInformer := informers.NewPVInformer(fakeKubeClient, resyncPeriod)

	// Act
	_, err := NewSnapshotController(
		fakeKubeClient,
		pvcInformer,
		pvInformer,
		nil, /* cloud */
		nil /* plugins */)

	// Assert
	if err != nil {
		t.Fatalf("Run failed with error. Expected: <no error> Actual: <%v>", err)
	}
}

/*
Copyright 2015 Google Inc. All rights reserved.

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

package kubelet

import (
	"testing"
)

func TestParsePodFullName(t *testing.T) {
	// Arrange
	podFullName := "ca4e7148-9ab9-11e4-924c-f0921cde18c1.default.etcd"

	// Act
	podName, podNamespace, podAnnotations := ParsePodFullName(podFullName)

	// Assert
	expectedPodName := "ca4e7148-9ab9-11e4-924c-f0921cde18c1"
	expectedPodNamespace := "default"
	expectedSource := "etcd"
	if podName != expectedPodName {
		t.Errorf("Unexpected PodName. Expected: %q Actual: %q", expectedPodName, podName)
	}
	if podNamespace != expectedPodNamespace {
		t.Errorf("Unexpected PodNamespace. Expected: %q Actual: %q", expectedPodNamespace, podNamespace)
	}
	if podAnnotations[ConfigSourceAnnotationKey] != expectedSource {
		t.Errorf("Unexpected PodSource. Expected: %q Actual: %q", expectedPodNamespace, podNamespace)
	}

}

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

package rook

import (
	"testing"
)

func TestGenerateTPRName(t *testing.T) {

	expectedTPR := "attach-namespace-mygroup-id-localhost"
	tpr := generateTPRName("namespace", "mygroup", "id", "localhost")

	if expectedTPR != tpr {
		t.Errorf("generateTPRName error: expected %s, got %s", expectedTPR, tpr)
	}
}

func TestGenerateTPRNameLowerCase(t *testing.T) {

	expectedTPR := "attach-namespace-mygroup-id-localhost"
	tpr := generateTPRName("NamEsPaCe", "mYGrOUP", "ID", "LoCALHost")

	if expectedTPR != tpr {
		t.Errorf("generateTPRName error: expected %s, got %s", expectedTPR, tpr)
	}
}

func TestGenerateTPRNameReplaceDot(t *testing.T) {

	expectedTPR := "attach-namespace-mygroup-id-127-0-0-1"
	tpr := generateTPRName("namespace", "mygroup", "id", "127.0.0.1")

	if expectedTPR != tpr {
		t.Errorf("generateTPRName error: expected %s, got %s", expectedTPR, tpr)
	}
}

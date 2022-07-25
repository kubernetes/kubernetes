/*
Copyright 2020 The Kubernetes Authors.

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

package fake

import (
	"bytes"
	"context"
	"io"
	"testing"

	corev1 "k8s.io/api/core/v1"
	cgtesting "k8s.io/client-go/testing"
)

func TestFakePodsGetLogs(t *testing.T) {
	fp := FakePods{
		Fake: &FakeCoreV1{Fake: &cgtesting.Fake{}},
		ns:   "default",
	}
	req := fp.GetLogs("foo", &corev1.PodLogOptions{})
	body, err := req.Stream(context.Background())
	if err != nil {
		t.Fatal("Stream pod logs:", err)
	}
	var buf bytes.Buffer
	n, err := io.Copy(&buf, body)
	if err != nil {
		t.Fatal("Read pod logs:", err)
	}
	if n == 0 {
		t.Fatal("Empty log")
	}
	err = body.Close()
	if err != nil {
		t.Fatal("Close response body:", err)
	}
}

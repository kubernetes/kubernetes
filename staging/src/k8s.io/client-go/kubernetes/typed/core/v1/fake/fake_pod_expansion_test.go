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

	testData := []struct {
		pod       string
		container string
		logs      string
	}{
		{
			pod:       "foo",
			logs:      defaultFakeLogs,
		},
		{
			pod:       "foo",
			container: "sidecar",
			logs:      "custom log from user",
		},
	}

	for i := range testData {
		d := testData[i]
		if d.container != "" && d.logs != defaultFakeLogs {
			if err := fp.SetContainerFakeLogs(d.container, d.logs); err != nil {
				t.Fatal("Set container logs:", err)
			}
		}
		req := fp.GetLogs(d.pod, &corev1.PodLogOptions{
			Container: d.container,
		})
		body, err := req.Stream(context.Background())
		if err != nil {
			t.Fatal("Stream pod logs:", err)
		}
		var buf bytes.Buffer
		n, err := io.Copy(&buf, body)
		if err != nil {
			t.Fatal("Read pod logs:", err)
		}
		if int64(len(d.logs)) != n || buf.String() != d.logs {
			t.Fatal("Fake logs mismatch")
		}
		err = body.Close()
		if err != nil {
			t.Fatal("Close response body:", err)
		}
	}
}

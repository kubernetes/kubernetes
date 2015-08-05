/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package cmd

import (
	"bytes"
	"net/http"
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestDiff(t *testing.T) {
	_, svc, _ := testData()
	svc.Items[0].Name = "redis-master"
	svc.Items[0].ObjectMeta = api.ObjectMeta{
		Name:      "redis-master",
		Namespace: "tester",
		Labels: map[string]string{
			"name": "redis-masterz",
		},
	}
	svc.Items[0].Spec = api.ServiceSpec{
		Ports: []api.ServicePort{
			{
				Port:       6378,
				TargetPort: util.NewIntOrStringFromInt(6379),
				Protocol:   "TCP",
			},
		},
		Selector: map[string]string{
			"name": "redis-master",
		},
	}

	f, tf, codec := NewAPIFactory()
	tf.Printer = &testPrinter{}
	tf.Client = &client.FakeRESTClient{
		Codec: codec,
		Resp:  &http.Response{StatusCode: 200, Body: objBody(codec, &svc.Items[0])},
	}
	tf.Namespace = "test"
	buf := bytes.NewBuffer([]byte{})

	cmd := NewCmdDiff(f, buf)
	cmd.SetOutput(buf)
	cmd.Flags().Set("filename", "../../../examples/guestbook/redis-master-service.yaml")
	cmd.Run(cmd, []string{})

	expectedDiffs := []string{
		"-    \"namespace\": test",
		"+    \"namespace\": tester",
		"-      \"name\": redis-master",
		"+      \"name\": redis-masterz",
	}

	if len(buf.String()) == 0 {
		t.Errorf("unexpected empty output")
	}
	if isPresent, diffLine := verifyDiffLines(buf.String(), expectedDiffs); !isPresent {
		t.Errorf("Unexpected output:\n'%s' cannot be found in the diff:\n%s\n", diffLine, buf.String())
	}
}

// Verifies that the diff contains all expected diff lines. If expected line is
// returned if it is missing.
func verifyDiffLines(diff string, expectedDiffs []string) (bool, string) {
	for _, s := range expectedDiffs {
		if !strings.Contains(diff, s) {
			return false, s
		}
	}
	return true, ""
}

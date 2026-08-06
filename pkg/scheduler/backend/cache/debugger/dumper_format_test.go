/*
Copyright 2026 The Kubernetes Authors.

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

package debugger

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"go.uber.org/zap/zapcore"
	v1 "k8s.io/api/core/v1"
	componentbasejson "k8s.io/component-base/logs/json"
	"k8s.io/klog/v2/textlogger"
)

func sampleNodes() nodeDumps {
	return nodeDumps{{
		Name:        "node-1",
		Deleted:     false,
		Requested:   resourceDump{MilliCPU: 1500, Memory: 2147483648},
		Allocatable: resourceDump{MilliCPU: 4000, Memory: 8589934592},
		Pods: podDumps{
			{Name: "web-0", Namespace: "default", UID: "a1b2", Phase: v1.PodRunning},
		},
		NominatedPods: podDumps{
			{Name: "batch-9", Namespace: "default", UID: "e5f6", Phase: v1.PodPending, NominatedNode: "node-1"},
		},
	}}
}

// TestDumpNodesTextFormat verifies the text backend renders nodes as the
// readable multi-line value (via String), using klog's line-continuation.
func TestDumpNodesTextFormat(t *testing.T) {
	var buf bytes.Buffer
	logger := textlogger.NewLogger(textlogger.NewConfig(textlogger.Output(&buf)))
	logger.Info("Dump of cached NodeInfo", "nodes", sampleNodes())

	out := buf.String()
	t.Logf("text output:\n%s", out)
	for _, want := range []string{
		"nodes=<",           // multi-line value delimiter
		"Node name: node-1", // human-readable field
		"name: web-0",       // scheduled pod
		"Nominated Pods",    // nominated section
	} {
		if !strings.Contains(out, want) {
			t.Errorf("text output missing %q", want)
		}
	}
	// Must NOT be a flattened single-line escaped string.
	if strings.Contains(out, `nodes="`) {
		t.Errorf("nodes was emitted as a quoted string, expected multi-line block:\n%s", out)
	}
}

// TestDumpNodesJSONFormat verifies the JSON backend (zap/zapr) encodes nodes as
// a nested array of structs (via MarshalLog), not as an escaped string.
func TestDumpNodesJSONFormat(t *testing.T) {
	var buf bytes.Buffer
	logger, _ := componentbasejson.NewJSONLogger(0, zapcore.AddSync(&buf), nil, nil)
	logger.Info("Dump of cached NodeInfo", "nodes", sampleNodes())

	out := buf.Bytes()
	t.Logf("json output:\n%s", out)

	var entry struct {
		Nodes []struct {
			Name        string `json:"name"`
			Deleted     bool   `json:"deleted"`
			Allocatable struct {
				MilliCPU int64 `json:"milliCPU"`
			} `json:"allocatable"`
			Pods []struct {
				Name string `json:"name"`
			} `json:"pods"`
		} `json:"nodes"`
	}
	if err := json.Unmarshal(out, &entry); err != nil {
		t.Fatalf("output is not a JSON object with a structured nodes field (got escaped string?): %v\n%s", err, out)
	}
	if len(entry.Nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(entry.Nodes))
	}
	n := entry.Nodes[0]
	if n.Name != "node-1" || n.Allocatable.MilliCPU != 4000 || len(n.Pods) != 1 || n.Pods[0].Name != "web-0" {
		t.Errorf("nodes not encoded as queryable struct: %+v", n)
	}
}

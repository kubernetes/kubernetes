/*
Copyright 2022 The Kubernetes Authors.

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

package logs_test

import (
	"os"
	"path"
	"testing"

	"k8s.io/component-base/logs"
	logsapi "k8s.io/component-base/logs/api/v1"
	_ "k8s.io/component-base/logs/json/register"
	"k8s.io/klog/v2"
)

func TestGlogSetter(t *testing.T) {
	testcases := map[string]struct {
		init func(t *testing.T)
		// write must write at verbosity level 1.
		write func()
	}{
		"klog": {
			init: func(t *testing.T) {},
			write: func() {
				klog.V(1).Info("hello")
			},
		},
		"json": {
			init: func(t *testing.T) {
				c := logsapi.NewLoggingConfiguration()
				c.Format = "json"
				if err := logsapi.ValidateAndApply(c, nil /* feature gates */); err != nil {
					t.Fatalf("Unexpected error enabling json output: %v", err)
				}
			},
			write: func() {
				klog.Background().V(1).Info("hello")
			},
		},
		"text": {
			init: func(t *testing.T) {
				c := logsapi.NewLoggingConfiguration()
				c.Format = "text"
				if err := logsapi.ValidateAndApply(c, nil /* feature gates */); err != nil {
					t.Fatalf("Unexpected error enabling text output: %v", err)
				}
			},
			write: func() {
				klog.Background().V(1).Info("hello")
			},
		},
	}

	for name, tc := range testcases {
		t.Run(name, func(t *testing.T) {
			state := klog.CaptureState()
			t.Cleanup(state.Restore)
			tmpdir := t.TempDir()
			tmpfile := path.Join(tmpdir, "stderr.log")
			oldStderr := os.Stderr
			defer func() {
				os.Stderr = oldStderr
			}()
			newStderr, err := os.Create(tmpfile)
			if err != nil {
				t.Fatalf("Unexpected error creating temp file: %v", err)
			}
			os.Stderr = newStderr

			defer func() {
				if err := logsapi.ResetForTest(nil /* feature gates */); err != nil {
					t.Errorf("Unexpected error resetting the logging configuration: %v", err)
				}
			}()
			tc.init(t)

			// First write with default verbosity level of 0 -> no output.
			tc.write()
			klog.Flush()
			out, err := os.ReadFile(tmpfile)
			if err != nil {
				t.Fatalf("Unexpected error reading temp file: %v", err)
			}
			if len(out) > 0 {
				t.Fatalf("Info message should have been discarded, got instead:\n%s", string(out))
			}

			// Increase verbosity at runtime.
			if _, err := logs.GlogSetter("1"); err != nil {
				t.Fatalf("Unexpected error setting verbosity level: %v", err)
			}

			// Now write again -> output.
			tc.write()
			klog.Flush()
			out, err = os.ReadFile(tmpfile)
			if err != nil {
				t.Fatalf("Unexpected error reading temp file: %v", err)
			}
			if len(out) == 0 {
				t.Fatal("Info message should have been written, got empty file instead.")
			}
		})
	}
}

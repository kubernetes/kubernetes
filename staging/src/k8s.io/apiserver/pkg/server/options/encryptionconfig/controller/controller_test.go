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

package controller

import (
	"context"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestProcessEncryptionConfig(t *testing.T) {
	testCases := []struct {
		name        string
		filePath    string
		expectError bool
	}{
		{
			name:        "empty config file",
			filePath:    "testdata/empty_config.yaml",
			expectError: true,
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			ctx := context.Background()
			d := NewDynamicKMSEncryptionConfiguration(
				testCase.name,
				testCase.filePath,
				nil,
				"",
				ctx.Done(),
			)

			_, _, err := d.processEncryptionConfig(ctx)
			if testCase.expectError && err == nil {
				t.Fatalf("expected error but got none")
			}
			if !testCase.expectError && err != nil {
				t.Fatalf("expected no error but got %v", err)
			}
		})
	}
}

func TestWatchEncryptionConfigFile(t *testing.T) {
	testCases := []struct {
		name          string
		generateEvent func(filePath string, cancel context.CancelFunc)
		expectError   bool
	}{
		{
			name:        "file not renamed or removed",
			expectError: false,
			generateEvent: func(filePath string, cancel context.CancelFunc) {
				os.Chtimes(filePath, time.Now(), time.Now())

				// wait for the event to be handled
				time.Sleep(1 * time.Second)
				cancel()
				os.Remove(filePath)
			},
		},
		{
			name:        "file renamed",
			expectError: true,
			generateEvent: func(filePath string, cancel context.CancelFunc) {
				os.Rename(filePath, filePath+"1")

				// wait for the event to be handled
				time.Sleep(1 * time.Second)
				os.Remove(filePath + "1")
			},
		},
		{
			name:        "file removed",
			expectError: true,
			generateEvent: func(filePath string, cancel context.CancelFunc) {
				// allow watcher handle to start
				time.Sleep(1 * time.Second)
				os.Remove(filePath)
			},
		},
	}

	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			testFilePath := copyFileForTest(t, "testdata/ec_config.yaml")

			d := NewDynamicKMSEncryptionConfiguration(
				testCase.name,
				testFilePath,
				nil,
				"",
				ctx.Done(),
			)

			errs := make(chan error, 1)
			go func() {
				err := d.watchEncryptionConfigFile(d.stopCh)
				errs <- err
			}()

			testCase.generateEvent(d.filePath, cancel)

			err := <-errs
			if testCase.expectError && err == nil {
				t.Fatalf("expected error but got none")
			}
			if !testCase.expectError && err != nil {
				t.Fatalf("expected no error but got %v", err)
			}
		})
	}
}

func copyFileForTest(t *testing.T, srcFilePath string) string {
	t.Helper()

	// get directory from source file path
	srcDir := filepath.Dir(srcFilePath)

	// get file name from source file path
	srcFileName := filepath.Base(srcFilePath)

	// set new file path
	dstFilePath := filepath.Join(srcDir, "test_"+srcFileName)

	// copy src file to dst file
	r, err := os.Open(srcFilePath)
	if err != nil {
		t.Fatalf("failed to open source file: %v", err)
	}
	defer r.Close()

	w, err := os.Create(dstFilePath)
	if err != nil {
		t.Fatalf("failed to create destination file: %v", err)
	}
	defer w.Close()

	// copy the file
	_, err = io.Copy(w, r)
	if err != nil {
		t.Fatalf("failed to copy file: %v", err)
	}

	err = w.Close()
	if err != nil {
		t.Fatalf("failed to close destination file: %v", err)
	}

	return dstFilePath
}

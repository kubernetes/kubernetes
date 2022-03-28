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

package volumepathhandler

import (
	"crypto/rand"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	utiltesting "k8s.io/client-go/util/testing"
)

func TestDetachFileDevice(t *testing.T) {
	blockVolumePathHandler := NewBlockVolumePathHandler()
	retryInterval := 100 * time.Millisecond
	retryCount := 5

	tests := []struct {
		description string
		setupFunc   func(path string) (string, string, error)
		assertFunc  func(blockDevicePath string) error
	}{
		{
			description: "detach loopback device corresponding to an existing backing file",
			setupFunc: func(path string) (string, string, error) {
				filePath := filepath.Join(path, "file")
				if err := createFile(filePath, 1024*1024); err != nil {
					return "", "", err
				}
				blockDevicePath, err := makeLoopDevice(filePath)
				if err != nil {
					return "", "", err
				}
				return filePath, blockDevicePath, nil

			},
			assertFunc: func(blockDevicePath string) error {
				return verifyBlockDeviceRemoval(blockDevicePath, retryInterval, retryCount)
			},
		},
		{
			description: "detach loopback device corresponding to non existing backing file",
			setupFunc: func(path string) (string, string, error) {
				filePath := filepath.Join(path, "file")
				if err := createFile(filePath, 1024*1024); err != nil {
					return "", "", err
				}
				blockDevicePath, err := makeLoopDevice(filePath)
				if err != nil {
					return "", "", err
				}
				// Simulate an unexpected detachment and reattachment of the backing file
				e := os.Remove(filePath)
				if e != nil {
					return "", "", err
				}
				if err := createFile(filePath, 512*1024); err != nil {
					return "", "", err
				}
				return filePath, blockDevicePath, nil

			},
			assertFunc: func(blockDevicePath string) error {
				return verifyBlockDeviceRemoval(blockDevicePath, retryInterval, retryCount)
			},
		},
	}

	for _, test := range tests {
		t.Run(test.description, func(t *testing.T) {
			tmpDir, err := utiltesting.MkTmpdir("volume_path_handler_linux")
			if err != nil {
				t.Fatalf("error creating temp dir: %v", err)
			}
			var blockDeviceToCleanUp []string

			defer func() {
				for _, blockDevicePath := range blockDeviceToCleanUp {
					removeLoopDevice(blockDevicePath)
				}
				os.RemoveAll(tmpDir)
			}()

			filePath, blockDevicePath, err := test.setupFunc(tmpDir)
			if err != nil {
				t.Fatalf("for %s error running setup with: %v", test.description, err)
			}
			blockDeviceToCleanUp = append(blockDeviceToCleanUp, blockDevicePath)

			err = blockVolumePathHandler.DetachFileDevice(filePath)
			if err != nil {
				t.Fatalf("for %s error detaching device with: %v", test.description, err)
			}

			err = test.assertFunc(blockDevicePath)
			if err != nil {
				t.Fatalf("for %s error verifying removing device: %v", test.description, err)
			}
		})
	}
}

func createFile(filePath string, size int64) error {
	file, err := os.OpenFile(filePath, os.O_RDWR|os.O_CREATE, 0755)
	if err != nil {
		return err
	}
	_, err = io.CopyN(file, rand.Reader, size)
	if err != nil {
		return err
	}
	file.Close()
	return nil
}

func verifyBlockDeviceRemoval(blockDevicePath string, retryInterval time.Duration, retryCount int) error {
	args := []string{blockDevicePath}
	for i := 0; i < retryCount; i++ {
		time.Sleep(retryInterval)
		cmd := exec.Command(losetupPath, args...)
		out, err := cmd.CombinedOutput()
		if strings.Contains(string(out), "No such file or directory") {
			return nil
		}
		fmt.Printf("couldn't verify the removed block device: %v: out: %v, err: %v \n", cmd.String(), string(out), err)
		fmt.Printf("retrying in %v \n", retryInterval)
	}
	return fmt.Errorf("failed to verify that loopback device %v is removed", blockDevicePath)
}

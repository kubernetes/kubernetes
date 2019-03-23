// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package devicemapper

import (
	"bufio"
	"bytes"
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"k8s.io/klog"
)

// thinLsClient knows how to run a thin_ls very specific to CoW usage for
// containers.
type thinLsClient interface {
	// ThinLs runs a thin ls on the given device, which is expected to be a
	// metadata device. The caller must hold the metadata snapshot for the
	// device.
	ThinLs(deviceName string) (map[string]uint64, error)
}

// newThinLsClient returns a thinLsClient or an error if the thin_ls binary
// couldn't be located.
func newThinLsClient() (thinLsClient, error) {
	thinLsPath, err := ThinLsBinaryPresent()
	if err != nil {
		return nil, fmt.Errorf("error creating thin_ls client: %v", err)
	}

	return &defaultThinLsClient{thinLsPath}, nil
}

// defaultThinLsClient is a functional thinLsClient
type defaultThinLsClient struct {
	thinLsPath string
}

var _ thinLsClient = &defaultThinLsClient{}

func (c *defaultThinLsClient) ThinLs(deviceName string) (map[string]uint64, error) {
	args := []string{"--no-headers", "-m", "-o", "DEV,EXCLUSIVE_BYTES", deviceName}
	klog.V(4).Infof("running command: thin_ls %v", strings.Join(args, " "))

	output, err := exec.Command(c.thinLsPath, args...).Output()
	if err != nil {
		return nil, fmt.Errorf("Error running command `thin_ls %v`: %v\noutput:\n\n%v", strings.Join(args, " "), err, string(output))
	}

	return parseThinLsOutput(output), nil
}

// parseThinLsOutput parses the output returned by thin_ls to build a map of
// device id -> usage.
func parseThinLsOutput(output []byte) map[string]uint64 {
	cache := map[string]uint64{}

	// parse output
	scanner := bufio.NewScanner(bytes.NewReader(output))
	for scanner.Scan() {
		output := scanner.Text()
		fields := strings.Fields(output)
		if len(fields) != 2 {
			continue
		}

		deviceID := fields[0]
		usage, err := strconv.ParseUint(fields[1], 10, 64)
		if err != nil {
			klog.Warningf("unexpected error parsing thin_ls output: %v", err)
			continue
		}

		cache[deviceID] = usage
	}

	return cache

}

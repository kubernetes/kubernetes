// +build darwin

/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package util

import (
	"errors"
	"fmt"
	"os/exec"
	"strings"

	"k8s.io/kubernetes/pkg/api/resource"
)

// FSInfo linux returns (available bytes, byte capacity, error) for the filesystem that
// path resides upon.
func FsInfo(path string) (int64, int64, error) {
	return 0, 0, errors.New("FsInfo not supported for this build.")
}

func Du(path string) (*resource.Quantity, error) {
	out, err := exec.Command("nice", "-n", "19", "du", "-s", path).CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("failed command 'du' ($ nice -n 19 du -s) on path %s with error %v", path, err)
	}
	used, err := resource.ParseQuantity(strings.Fields(string(out))[0])
	if err != nil {
		return nil, fmt.Errorf("failed to parse 'du' output %s due to error %v", out, err)
	}
	used.Format = resource.BinarySI
	return used, nil
}

/*
Copyright 2023 The Kubernetes Authors.

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

package golang

import (
	"errors"
	"k8s.io/klog/v2"
	"os"
	"os/exec"
	"strings"
)

// PackageOf returns the package name of the given path.
//
// TODO: Consider rewriting this to use go/packages instead of shelling out.
func PackageOf(path string) (string, error) {
	c := exec.Command("go", "list", "-find", path)
	c.Env = append(os.Environ(), "GO111MODULE=on")
	klog.V(3).Infof("Running: %q", c)
	out, err := c.Output()
	if err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			klog.Errorf("go list stderr: %s", ee.Stderr)
		}
		return "", err
	}
	return strings.Trim(string(out), "\n"), nil
}

/*
Copyright 2018 The Kubernetes Authors.

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

package system

import (
	"fmt"
	"strings"

	"github.com/golang/glog"
	selinux "k8s.io/kubernetes/pkg/util/selinux"
)

type SELinuxContext struct {
	User  string
	Role  string
	Type  string
	Level string
}

// Create a new SELinux context by default allowing files to be shared
// with containers
func NewSELinuxContext() *SELinuxContext {
	return &SELinuxContext{
		"unconfined_u", "object_r", "container_file_t", "s0"}
}

// Converts the context into an SELinux label
func (c SELinuxContext) ToString() string {
	arr := []string{c.User, c.Role, c.Type, c.Level}
	return strings.Join(arr, ":")
}

// Sets the SELinux file context using the provided SELinux context
func SetSELinuxFilecon(path string, context *SELinuxContext) error {
	if !selinux.SELinuxEnabled() {
		return nil
	}
	selinuxRunner := selinux.NewSELinuxRunner()
	err := selinuxRunner.Setfilecon(path, context.ToString())
	if err != nil {
		return fmt.Errorf("unable to set SELinux file context for file %q: [%v]", path, err)
	}
	glog.V(1).Infof("[selinux] Set SELinux file context on %q to %q.\n", path, context.ToString())
	return nil
}

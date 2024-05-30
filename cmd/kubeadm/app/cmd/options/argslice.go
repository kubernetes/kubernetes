/*
Copyright 2024 The Kubernetes Authors.

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

package options

import (
	"fmt"
	"strings"

	"github.com/pkg/errors"

	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
)

type argSlice struct {
	args *[]kubeadmapiv1.Arg
}

func newArgSlice(args *[]kubeadmapiv1.Arg) *argSlice {
	return &argSlice{args: args}
}

// String implements github.com/spf13/pflag.Value
func (s *argSlice) String() string {
	if s == nil || s.args == nil || len(*s.args) == 0 {
		return ""
	}

	pairs := make([]string, 0, len(*s.args))
	for _, a := range *s.args {
		pairs = append(pairs, fmt.Sprintf("%s=%s", a.Name, a.Value))
	}

	return strings.Join(pairs, ",")
}

// Set implements github.com/spf13/pflag.Value
func (s *argSlice) Set(value string) error {
	if s.args == nil {
		s.args = &[]kubeadmapiv1.Arg{}
	}

	pairs := strings.Split(value, ",")

	for _, p := range pairs {
		m := strings.Split(p, "=")
		if len(m) != 2 {
			return errors.Errorf("malformed key=value pair in flag value: %s", value)
		}
		arg := kubeadmapiv1.Arg{Name: m[0], Value: m[1]}
		*s.args = append(*s.args, arg)
	}

	return nil
}

// Type implements github.com/spf13/pflag.Value
func (s *argSlice) Type() string {
	return "[]kubeadmapiv1.Arg"
}

/*
Copyright 2016 The Kubernetes Authors.

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

// package taints implements uitilites for working with taints
package taints

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/util/validation"
)

// ParseTaint parses a taint from a string. Taint must be off the format '<key>=<value>:<effect>'.
func ParseTaint(st string) (api.Taint, error) {
	var taint api.Taint
	parts := strings.Split(st, "=")
	if len(parts) != 2 || len(parts[1]) == 0 || len(validation.IsQualifiedName(parts[0])) > 0 {
		return taint, fmt.Errorf("invalid taint spec: %v", st)
	}

	parts2 := strings.Split(parts[1], ":")

	effect := api.TaintEffect(parts2[1])

	errs := validation.IsValidLabelValue(parts2[0])
	if len(parts2) != 2 || len(errs) != 0 {
		return taint, fmt.Errorf("invalid taint spec: %v, %s", st, strings.Join(errs, "; "))
	}

	if effect != api.TaintEffectNoSchedule && effect != api.TaintEffectPreferNoSchedule {
		return taint, fmt.Errorf("invalid taint spec: %v, unsupported taint effect", st)
	}

	taint.Key = parts[0]
	taint.Value = parts2[0]
	taint.Effect = effect

	return taint, nil
}

// NewTaintsVar wraps []api.Taint in a struct that implements flag.Value to allow taints to be
// bound to command line flags.
func NewTaintsVar(ptr *[]api.Taint) taintsVar {
	return taintsVar{
		ptr: ptr,
	}
}

type taintsVar struct {
	ptr *[]api.Taint
}

func (t taintsVar) Set(s string) error {
	sts := strings.Split(s, ",")
	var taints []api.Taint
	for _, st := range sts {
		taint, err := ParseTaint(st)
		if err != nil {
			return err
		}
		taints = append(taints, taint)
	}
	*t.ptr = taints
	return nil
}

func (t taintsVar) String() string {
	if len(*t.ptr) == 0 {
		return "<nil>"
	}
	var taints []string
	for _, taint := range *t.ptr {
		taints = append(taints, fmt.Sprintf("%s=%s:%s", taint.Key, taint.Value, taint.Effect))
	}
	return strings.Join(taints, ",")
}

func (t taintsVar) Type() string {
	return "[]api.Taint"
}

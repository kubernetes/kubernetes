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

// package taints implements utilities for working with taints
package taints

import (
	"fmt"
	"strings"

	utiltaints "k8s.io/cloud-provider/util/taints"
	api "k8s.io/kubernetes/pkg/apis/core"
)

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
	if len(s) == 0 {
		*t.ptr = nil
		return nil
	}
	sts := strings.Split(s, ",")
	var taints []api.Taint
	for _, st := range sts {
		taint, err := utiltaints.ParseTaint(st)
		if err != nil {
			return err
		}
		taints = append(taints, api.Taint{Key: taint.Key, Value: taint.Value, Effect: api.TaintEffect(taint.Effect)})
	}
	*t.ptr = taints
	return nil
}

func (t taintsVar) String() string {
	if len(*t.ptr) == 0 {
		return ""
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

/*
Copyright 2019 The Kubernetes Authors.

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

package v1

import (
	"k8s.io/apimachinery/pkg/runtime"
)

func (in *TableRow) DeepCopy() *TableRow {
	if in == nil {
		return nil
	}

	out := new(TableRow)

	if in.Cells != nil {
		out.Cells = make([]interface{}, len(in.Cells))
		for i := range in.Cells {
			out.Cells[i] = runtime.DeepCopyJSONValue(in.Cells[i])
		}
	}

	if in.Conditions != nil {
		out.Conditions = make([]TableRowCondition, len(in.Conditions))
		for i := range in.Conditions {
			in.Conditions[i].DeepCopyInto(&out.Conditions[i])
		}
	}

	in.Object.DeepCopyInto(&out.Object)
	return out
}

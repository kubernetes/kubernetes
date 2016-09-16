/*
Copyright 2014 The Kubernetes Authors.

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

package aws

import (
	"github.com/aws/aws-sdk-go/aws"
	"k8s.io/kubernetes/pkg/util/sets"
)

func stringSetToPointers(in sets.String) []*string {
	if in == nil {
		return nil
	}
	out := make([]*string, 0, len(in))
	for k := range in {
		out = append(out, aws.String(k))
	}
	return out
}

func stringSetFromPointers(in []*string) sets.String {
	if in == nil {
		return nil
	}
	out := sets.NewString()
	for i := range in {
		out.Insert(orEmpty(in[i]))
	}
	return out
}

func orZero(v *int64) int64 {
	if v == nil {
		return 0
	}
	return *v
}

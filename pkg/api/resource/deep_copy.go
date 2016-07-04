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

package resource

import (
	inf "gopkg.in/inf.v0"

	conversion "k8s.io/kubernetes/pkg/conversion"
)

func DeepCopy_resource_Quantity(in Quantity, out *Quantity, c *conversion.Cloner) error {
	*out = in
	if in.d.Dec != nil {
		tmp := &inf.Dec{}
		out.d.Dec = tmp.Set(in.d.Dec)
	}
	return nil
}

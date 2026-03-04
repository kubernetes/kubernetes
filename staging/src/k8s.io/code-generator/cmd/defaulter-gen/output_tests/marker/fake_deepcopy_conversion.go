/*
Copyright 2020 The Kubernetes Authors.

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

package marker

import (
	runtime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func (in *Defaulted) DeepCopy() *Defaulted {
	if in == nil {
		return nil
	}
	out := new(Defaulted)
	in.DeepCopyInto(out)
	return out
}

func (in *Defaulted) DeepCopyInto(out *Defaulted) {
	*out = *in
}

func (in *Defaulted) DeepCopyObject() runtime.Object {
	if c := in.DeepCopy(); c != nil {
		return c
	}
	return nil
}

func (in *DefaultedOmitempty) DeepCopy() *DefaultedOmitempty {
	if in == nil {
		return nil
	}
	out := new(DefaultedOmitempty)
	in.DeepCopyInto(out)
	return out
}

func (in *DefaultedOmitempty) DeepCopyInto(out *DefaultedOmitempty) {
	*out = *in
}

func (in *DefaultedOmitempty) DeepCopyObject() runtime.Object {
	if c := in.DeepCopy(); c != nil {
		return c
	}
	return nil
}

func (in *DefaultedWithFunction) DeepCopy() *DefaultedWithFunction {
	if in == nil {
		return nil
	}
	out := new(DefaultedWithFunction)
	in.DeepCopyInto(out)
	return out
}

func (in *DefaultedWithFunction) DeepCopyInto(out *DefaultedWithFunction) {
	*out = *in
}

func (in *DefaultedWithFunction) DeepCopyObject() runtime.Object {
	if c := in.DeepCopy(); c != nil {
		return c
	}
	return nil
}

func (in *Defaulted) GetObjectKind() schema.ObjectKind              { return schema.EmptyObjectKind }
func (in *DefaultedOmitempty) GetObjectKind() schema.ObjectKind     { return schema.EmptyObjectKind }
func (in *DefaultedWithFunction) GetObjectKind() schema.ObjectKind  { return schema.EmptyObjectKind }
func (in *DefaultedWithReference) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }

func (in *DefaultedWithReference) DeepCopy() *DefaultedWithReference {
	if in == nil {
		return nil
	}
	out := new(DefaultedWithReference)
	in.DeepCopyInto(out)
	return out
}

func (in *DefaultedWithReference) DeepCopyInto(out *DefaultedWithReference) {
	*out = *in
}

func (in *DefaultedWithReference) DeepCopyObject() runtime.Object {
	if c := in.DeepCopy(); c != nil {
		return c
	}
	return nil
}

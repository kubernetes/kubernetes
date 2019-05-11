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

package admission

import "k8s.io/apimachinery/pkg/runtime"

type SchemeBasedObjectInterfaces struct {
	Scheme *runtime.Scheme
}

func (r *SchemeBasedObjectInterfaces) GetObjectCreater() runtime.ObjectCreater     { return r.Scheme }
func (r *SchemeBasedObjectInterfaces) GetObjectTyper() runtime.ObjectTyper         { return r.Scheme }
func (r *SchemeBasedObjectInterfaces) GetObjectDefaulter() runtime.ObjectDefaulter { return r.Scheme }
func (r *SchemeBasedObjectInterfaces) GetObjectConvertor() runtime.ObjectConvertor { return r.Scheme }

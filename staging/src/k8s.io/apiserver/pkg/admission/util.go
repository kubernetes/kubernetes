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

type RuntimeObjectInterfaces struct {
	runtime.ObjectCreater
	runtime.ObjectTyper
	runtime.ObjectDefaulter
	runtime.ObjectConvertor
	runtime.EquivalentResourceMapper
}

func NewObjectInterfacesFromScheme(scheme *runtime.Scheme) ObjectInterfaces {
	return &RuntimeObjectInterfaces{scheme, scheme, scheme, scheme, runtime.NewEquivalentResourceRegistry()}
}

func (r *RuntimeObjectInterfaces) GetObjectCreater() runtime.ObjectCreater {
	return r.ObjectCreater
}
func (r *RuntimeObjectInterfaces) GetObjectTyper() runtime.ObjectTyper {
	return r.ObjectTyper
}
func (r *RuntimeObjectInterfaces) GetObjectDefaulter() runtime.ObjectDefaulter {
	return r.ObjectDefaulter
}
func (r *RuntimeObjectInterfaces) GetObjectConvertor() runtime.ObjectConvertor {
	return r.ObjectConvertor
}
func (r *RuntimeObjectInterfaces) GetEquivalentResourceMapper() runtime.EquivalentResourceMapper {
	return r.EquivalentResourceMapper
}

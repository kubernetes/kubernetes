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

package util

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
)

const (
	// ServiceImportPrefix is used to identify imported services
	// and separate them from traditional clusters local services
	// that may have the same name. This prefix intentionally
	// makes a service name invalid to guarantee that there are
	// never collisions with same-named cluster local services.
	// The new service name may be used in log messages and other
	// user-facing strings.
	ServiceImportPrefix = "import:"
)

// ServiceImportName returns an import name for the supplied  service,
// formatted to avoid conflicts with same-named regular services. This
// name is not a valid K8s Service name and is for internal use only.
func ServiceImportName(serviceName string) string {
	return ServiceImportPrefix + serviceName
}

type importServiceWrapper struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              struct {
		v1.ServiceSpec `json:",inline"`
		IP             string `json:"ip,omitempty"`
	} `json:"spec,omitempty"`
}

// ServiceFromImportInformer takes an interface passed by an unstructured informer
// and attempts to convert it to a Service.
func ServiceFromImportInformer(obj interface{}) (*v1.Service, error) {
	resource, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return nil, fmt.Errorf("unexpected object type: %v", obj)
	}
	var serviceImport importServiceWrapper
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(resource.UnstructuredContent(), &serviceImport); err != nil {
		return nil, err
	}
	serviceImport.Name = ServiceImportName(serviceImport.Name)
	if serviceImport.Spec.Type == "Headless" {
		serviceImport.Spec.ClusterIP = "none"
	} else {
		serviceImport.Spec.ClusterIP = serviceImport.Spec.IP
	}
	serviceImport.Spec.Type = v1.ServiceTypeClusterIP
	return &v1.Service{
		ObjectMeta: serviceImport.ObjectMeta,
		Spec:       serviceImport.Spec.ServiceSpec,
	}, nil
}

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
	mcsv1alpha1 "k8s.io/mcs-api/pkg/apis/multicluster/v1alpha1"
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

// ServiceFromImport converts a ServiceImport to a Service resource
// for internal use.
func ServiceFromImport(svc *mcsv1alpha1.ServiceImport) *v1.Service {
	cp := make([]v1.ServicePort, len(svc.Spec.Ports))
	for i, p := range svc.Spec.Ports {
		cp[i] = v1.ServicePort{
			Name:        p.Name,
			Port:        p.Port,
			Protocol:    p.Protocol,
			AppProtocol: p.AppProtocol,
		}
	}
	clusterIP := svc.Spec.IP
	if svc.Spec.Type == mcsv1alpha1.Headless {
		clusterIP = "none"
	}
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: svc.Namespace,
			Name:      ServiceImportName(svc.Name),
		},
		Spec: v1.ServiceSpec{
			ClusterIP:             clusterIP,
			Ports:                 cp,
			Type:                  v1.ServiceTypeClusterIP,
			SessionAffinity:       svc.Spec.SessionAffinity,
			SessionAffinityConfig: svc.Spec.SessionAffinityConfig,
		},
	}
}

// ServiceImportFromInformer takes an interface passed by an unstructured informer
// and attempts to convert it to a ServiceImport.
func ServiceImportFromInformer(obj interface{}) (*mcsv1alpha1.ServiceImport, error) {
	resource, ok := obj.(*unstructured.Unstructured)
	if !ok {
		return nil, fmt.Errorf("unexpected object type: %v", obj)
	}
	var serviceImport mcsv1alpha1.ServiceImport
	if err := runtime.DefaultUnstructuredConverter.FromUnstructured(resource.UnstructuredContent(), &serviceImport); err != nil {
		return nil, err
	}
	return &serviceImport, nil
}

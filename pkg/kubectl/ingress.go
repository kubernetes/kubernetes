/*
Copyright 2017 The Kubernetes Authors.

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

package kubectl

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

// annotation name that says an ingress should get an ACME (Let's Encrypt) certificate
const ingressAnnotationNameTLSAcme = "kubernetes.io/tls-acme"

type IngressV1Beta1 struct {
	Name        string
	Host        []string
	ServiceName string
	ServicePort intstr.IntOrString
	TLSAcme     bool
}

func (IngressV1Beta1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"host", false},
		{"service-name", false},
		{"service-port", false},
		{"tls-acme", false},
	}
}

// Ensure it supports the generator pattern that uses parameter injection
var _ Generator = &IngressV1Beta1{}

// Ensure it supports the generator pattern that uses parameters specified during construction
var _ StructuredGenerator = &IngressV1Beta1{}

func (g IngressV1Beta1) Generate(params map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(g.ParamNames(), params)
	if err != nil {
		return nil, err
	}

	delegate := &IngressV1Beta1{}

	if name, isString := params["name"].(string); isString {
		delegate.Name = name
	} else {
		return nil, fmt.Errorf("expected string, saw %v for 'name'", params["name"])
	}

	if host, isArray := params["host"].([]string); isArray {
		delegate.Host = host
	} else {
		return nil, fmt.Errorf("expected []string, found %v for 'host'", params["host"])
	}

	if params["tls-acme"] != nil {
		if tlsAcme, isBool := params["tls-acme"].(bool); isBool {
			delegate.TLSAcme = tlsAcme
		} else {
			return nil, fmt.Errorf("expected bool, saw %v for 'tls-acme'", params["tls-acme"])
		}
	}

	if params["service-name"] != nil {
		if serviceName, isString := params["service-name"].(string); isString {
			delegate.ServiceName = serviceName
		} else {
			return nil, fmt.Errorf("expected string, saw %v for 'service-name'", params["service-name"])
		}
	}

	if params["service-port"] != nil {
		if servicePortInt, isInt := params["service-port"].(int); isInt {
			delegate.ServicePort = intstr.FromInt(servicePortInt)
		} else if servicePortString, isString := params["service-port"].(string); isString {
			delegate.ServicePort = intstr.FromString(servicePortString)
		} else {
			return nil, fmt.Errorf("expected string or int, saw %v for 'service-port'", params["service-port"])
		}
	}

	return delegate.StructuredGenerate()
}

// StructuredGenerate outputs an Ingress object using the configured fields
func (g *IngressV1Beta1) StructuredGenerate() (runtime.Object, error) {
	if err := g.validate(); err != nil {
		return nil, err
	}

	// setup default label and selector
	labels := map[string]string{}
	labels["app"] = g.Name

	ingress := extensions.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:   g.Name,
			Labels: labels,
		},
	}

	// Default backend service name to same name as ingress
	backendServiceName := g.ServiceName
	if backendServiceName == "" {
		backendServiceName = g.Name
	}

	// Default backend service port to 80
	backendServicePort := g.ServicePort
	if backendServicePort.IntVal == 0 && backendServicePort.StrVal == "" {
		backendServicePort.IntVal = 80
	}

	for _, host := range g.Host {
		rule := extensions.IngressRule{
			Host: host,
		}
		rule.HTTP = &extensions.HTTPIngressRuleValue{
			Paths: []extensions.HTTPIngressPath{
				{
					Path: "/",
					Backend: extensions.IngressBackend{
						ServiceName: backendServiceName,
						ServicePort: backendServicePort,
					},
				},
			},
		}
		ingress.Spec.Rules = append(ingress.Spec.Rules, rule)
	}

	if g.TLSAcme {
		tls := extensions.IngressTLS{
			Hosts:      g.Host,
			SecretName: "tls-" + g.Name,
		}
		ingress.Spec.TLS = append(ingress.Spec.TLS, tls)

		if ingress.Annotations == nil {
			ingress.Annotations = make(map[string]string)
		}
		ingress.Annotations[ingressAnnotationNameTLSAcme] = "true"
	}

	return &ingress, nil
}

// validate validates required fields are set to support structured generation
func (g *IngressV1Beta1) validate() error {
	if len(g.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if len(g.Host) == 0 {
		return fmt.Errorf("at least one host must be specified")
	}
	if g.ServicePort.StrVal != "" {
		errorMessages := validation.IsValidPortName(g.ServicePort.StrVal)
		if len(errorMessages) > 0 {
			return fmt.Errorf("invalid service-port name %s: %s", g.ServicePort.StrVal, errorMessages[0])
		}
	} else if g.ServicePort.IntVal != 0 {
		errorMessages := validation.IsValidPortNum(int(g.ServicePort.IntVal))
		if len(errorMessages) > 0 {
			return fmt.Errorf("invalid service-port number %d: %s", g.ServicePort.IntVal, errorMessages[0])
		}
	} else {
		// We will default the service port
	}

	return nil
}

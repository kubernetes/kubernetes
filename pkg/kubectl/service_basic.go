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

package kubectl

import (
	"fmt"
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/validation"
)

type ServiceCommonGeneratorV1 struct {
	Name         string
	TCP          []string
	Type         v1.ServiceType
	ClusterIP    string
	NodePort     int
	ExternalName string
}

type ServiceClusterIPGeneratorV1 struct {
	ServiceCommonGeneratorV1
}

type ServiceNodePortGeneratorV1 struct {
	ServiceCommonGeneratorV1
}

type ServiceLoadBalancerGeneratorV1 struct {
	ServiceCommonGeneratorV1
}

// TODO: is this really necessary?
type ServiceExternalNameGeneratorV1 struct {
	ServiceCommonGeneratorV1
}

func (ServiceClusterIPGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"tcp", true},
		{"clusterip", false},
	}
}
func (ServiceNodePortGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"tcp", true},
		{"nodeport", true},
	}
}
func (ServiceLoadBalancerGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"tcp", true},
	}
}

func (ServiceExternalNameGeneratorV1) ParamNames() []GeneratorParam {
	return []GeneratorParam{
		{"name", true},
		{"externalname", true},
	}
}

func parsePorts(portString string) (int32, intstr.IntOrString, error) {
	portStringSlice := strings.Split(portString, ":")

	port, err := strconv.Atoi(portStringSlice[0])
	if err != nil {
		return 0, intstr.FromInt(0), err
	}

	if errs := validation.IsValidPortNum(port); len(errs) != 0 {
		return 0, intstr.FromInt(0), fmt.Errorf(strings.Join(errs, ","))
	}

	if len(portStringSlice) == 1 {
		return int32(port), intstr.FromInt(int(port)), nil
	}

	var targetPort intstr.IntOrString
	if portNum, err := strconv.Atoi(portStringSlice[1]); err != nil {
		if errs := validation.IsValidPortName(portStringSlice[1]); len(errs) != 0 {
			return 0, intstr.FromInt(0), fmt.Errorf(strings.Join(errs, ","))
		}
		targetPort = intstr.FromString(portStringSlice[1])
	} else {
		if errs := validation.IsValidPortNum(portNum); len(errs) != 0 {
			return 0, intstr.FromInt(0), fmt.Errorf(strings.Join(errs, ","))
		}
		targetPort = intstr.FromInt(portNum)
	}
	return int32(port), targetPort, nil
}

func (s ServiceCommonGeneratorV1) GenerateCommon(params map[string]interface{}) error {
	name, isString := params["name"].(string)
	if !isString {
		return fmt.Errorf("expected string, saw %v for 'name'", name)
	}
	tcpStrings, isArray := params["tcp"].([]string)
	if !isArray {
		return fmt.Errorf("expected []string, found :%v", tcpStrings)
	}
	clusterip, isString := params["clusterip"].(string)
	if !isString {
		return fmt.Errorf("expected string, saw %v for 'clusterip'", clusterip)
	}
	externalname, isString := params["externalname"].(string)
	if !isString {
		return fmt.Errorf("expected string, saw %v for 'externalname'", externalname)
	}
	s.Name = name
	s.TCP = tcpStrings
	s.ClusterIP = clusterip
	s.ExternalName = externalname
	return nil
}

func (s ServiceLoadBalancerGeneratorV1) Generate(params map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), params)
	if err != nil {
		return nil, err
	}
	delegate := &ServiceCommonGeneratorV1{Type: v1.ServiceTypeLoadBalancer, ClusterIP: ""}
	err = delegate.GenerateCommon(params)
	if err != nil {
		return nil, err
	}
	return delegate.StructuredGenerate()
}

func (s ServiceNodePortGeneratorV1) Generate(params map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), params)
	if err != nil {
		return nil, err
	}
	delegate := &ServiceCommonGeneratorV1{Type: v1.ServiceTypeNodePort, ClusterIP: ""}
	err = delegate.GenerateCommon(params)
	if err != nil {
		return nil, err
	}
	return delegate.StructuredGenerate()
}

func (s ServiceClusterIPGeneratorV1) Generate(params map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), params)
	if err != nil {
		return nil, err
	}
	delegate := &ServiceCommonGeneratorV1{Type: v1.ServiceTypeClusterIP, ClusterIP: ""}
	err = delegate.GenerateCommon(params)
	if err != nil {
		return nil, err
	}
	return delegate.StructuredGenerate()
}

func (s ServiceExternalNameGeneratorV1) Generate(params map[string]interface{}) (runtime.Object, error) {
	err := ValidateParams(s.ParamNames(), params)
	if err != nil {
		return nil, err
	}
	delegate := &ServiceCommonGeneratorV1{Type: v1.ServiceTypeExternalName, ClusterIP: ""}
	err = delegate.GenerateCommon(params)
	if err != nil {
		return nil, err
	}
	return delegate.StructuredGenerate()
}

// validate validates required fields are set to support structured generation
// TODO(xiangpengzhao): validate ports are identity mapped for headless service when we enforce that in validation.validateServicePort.
func (s ServiceCommonGeneratorV1) validate() error {
	if len(s.Name) == 0 {
		return fmt.Errorf("name must be specified")
	}
	if len(s.Type) == 0 {
		return fmt.Errorf("type must be specified")
	}
	if s.ClusterIP == v1.ClusterIPNone && s.Type != v1.ServiceTypeClusterIP {
		return fmt.Errorf("ClusterIP=None can only be used with ClusterIP service type")
	}
	if s.ClusterIP != v1.ClusterIPNone && len(s.TCP) == 0 && s.Type != v1.ServiceTypeExternalName {
		return fmt.Errorf("at least one tcp port specifier must be provided")
	}
	if s.Type == v1.ServiceTypeExternalName {
		if errs := validation.IsDNS1123Subdomain(s.ExternalName); len(errs) != 0 {
			return fmt.Errorf("invalid service external name %s", s.ExternalName)
		}
	}
	return nil
}

func (s ServiceCommonGeneratorV1) StructuredGenerate() (runtime.Object, error) {
	err := s.validate()
	if err != nil {
		return nil, err
	}
	ports := []v1.ServicePort{}
	for _, tcpString := range s.TCP {
		port, targetPort, err := parsePorts(tcpString)
		if err != nil {
			return nil, err
		}

		portName := strings.Replace(tcpString, ":", "-", -1)
		ports = append(ports, v1.ServicePort{
			Name:       portName,
			Port:       port,
			TargetPort: targetPort,
			Protocol:   v1.Protocol("TCP"),
			NodePort:   int32(s.NodePort),
		})
	}

	// setup default label and selector
	labels := map[string]string{}
	labels["app"] = s.Name
	selector := map[string]string{}
	selector["app"] = s.Name

	service := v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:   s.Name,
			Labels: labels,
		},
		Spec: v1.ServiceSpec{
			Type:         v1.ServiceType(s.Type),
			Selector:     selector,
			Ports:        ports,
			ExternalName: s.ExternalName,
		},
	}
	if len(s.ClusterIP) > 0 {
		service.Spec.ClusterIP = s.ClusterIP
	}
	return &service, nil
}

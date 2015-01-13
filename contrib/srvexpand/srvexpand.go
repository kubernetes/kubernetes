/*
Copyright 2014 Google Inc. All rights reserved.

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

// srvexpand is a tool to generate non-trivial but regular services
// from a description free of most boilerplate
//
// $ srvexpand myservice.json | kubectl create -f -
// $ srvexpand myservice.yaml | kubectl create -f -
//
// This is completely separate from kubectl at the moment, until we figure out
// what the right integration approach is.
//
// Whether this type of wrapper should be encouraged is debatable. It eliminates
// some boilerplate, at the cost of needing to be updated whenever the generated
// API objects change. For instance, this initial version does not expose the
// protocol and createExternalLoadBalancer fields of Service. It's likely that we
// should support boilerplate elimination in the API itself, such as with more
// intelligent defaults, and generic transformations such as map keys to names.

package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strconv"
	"strings"

	// TODO: handle multiple versions correctly. Targeting v1beta3 because
	// v1beta1 is too much of a mess. Once we do support multiple versions,
	// it should be possible to specify the version for the whole map.
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/ghodss/yaml"
	"github.com/golang/glog"
)

const usage = "usage: srvexpand filename"

// Hierarchical service structures are a common pattern and allows omission
// of kind fields on input.

// TODO: Enable apiversion and namespace to be provided for the whole map.
// Note that I don't provide a way to specify labels and annotations to be
// propagated to all the objects (except those required to distinguish and
// connect the objects read in) because I expect that to be done as a
// separate pass.

type HierarchicalController struct {
	// Optional: Defaults to one
	Replicas int `json:"replicas,omitempty"`
	// Spec defines the behavior of a pod.
	Spec v1beta3.PodSpec `json:"spec,omitempty"`
}

type ControllerMap map[string]HierarchicalController

type HierarchicalService struct {
	// Optional: Creates a service if specified: servicePort:containerPort
	// TODO: Support multiple protocols
	PortSpec string `json:"portSpec,omitempty"`
	// Map of replication controllers to create
	ControllerMap ControllerMap `json:"controllers,omitempty"`
}

type ServiceMap map[string]HierarchicalService

func checkErr(err error) {
	if err != nil {
		glog.FatalDepth(1, err)
	}
}

func main() {
	if len(os.Args) != 2 {
		checkErr(fmt.Errorf(usage))
	}
	filename := os.Args[1]

	serviceMap := readServiceMap(filename)

	expandServiceMap(serviceMap)
}

func readServiceMap(filename string) ServiceMap {
	inData, err := ReadConfigData(filename)
	checkErr(err)

	serviceMap := ServiceMap{}
	err = yaml.Unmarshal(inData, &serviceMap)
	checkErr(err)

	return serviceMap
}

func canonicalizeName(name *string) {
	*name = strings.ToLower(*name)
	if !util.IsDNSLabel(*name) {
		checkErr(fmt.Errorf("name (%s) is not a valid DNS label", *name))
	}
}

func expandServiceMap(serviceMap ServiceMap) {
	for name, service := range serviceMap {
		canonicalizeName(&name)

		generateService(name, service.PortSpec)
		generateReplicationControllers(name, service.ControllerMap)
	}
}

func generateService(name string, portSpec string) {
	if portSpec == "" {
		return
	}

	servicePort, containerPort, err := portsFromString(portSpec)
	checkErr(err)

	svc := []v1beta3.Service{{
		TypeMeta: v1beta3.TypeMeta{APIVersion: "v1beta3", Kind: "Service"},
		ObjectMeta: v1beta3.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"service": name,
			},
		},
		Spec: v1beta3.ServiceSpec{
			Port:          servicePort,
			ContainerPort: util.NewIntOrStringFromInt(containerPort),
			Selector: map[string]string{
				"service": name,
			},
		},
	}}

	svcOutData, err := yaml.Marshal(svc)
	checkErr(err)

	fmt.Print(string(svcOutData))
}

func generateReplicationControllers(sname string, controllerMap ControllerMap) {
	for cname, controller := range controllerMap {
		canonicalizeName(&cname)

		generatePodTemplate(sname, cname, controller.Spec)
		generateReplicationController(sname, cname, controller.Replicas)
	}
}

func generatePodTemplate(sname string, cname string, podSpec v1beta3.PodSpec) {
	name := fmt.Sprintf("%s-%s", sname, cname)
	pt := []v1beta3.PodTemplate{{
		TypeMeta: v1beta3.TypeMeta{APIVersion: "v1beta3", Kind: "PodTemplate"},
		ObjectMeta: v1beta3.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"service": sname,
				"track":   cname,
			},
		},
		Spec: v1beta3.PodTemplateSpec{
			ObjectMeta: v1beta3.ObjectMeta{
				Labels: map[string]string{
					"service": sname,
					"track":   cname,
				},
			},
			Spec: podSpec,
		},
	}}

	ptOutData, err := yaml.Marshal(pt)
	checkErr(err)

	fmt.Print(string(ptOutData))
}

func generateReplicationController(sname string, cname string, replicas int) {
	if replicas < 1 {
		replicas = 1
	}

	name := fmt.Sprintf("%s-%s", sname, cname)
	rc := []v1beta3.ReplicationController{{
		TypeMeta: v1beta3.TypeMeta{APIVersion: "v1beta3", Kind: "ReplicationController"},
		ObjectMeta: v1beta3.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"service": sname,
				"track":   cname,
			},
		},
		Spec: v1beta3.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{
				"service": sname,
				"track":   cname,
			},
			Template: v1beta3.ObjectReference{
				Kind:       "PodTemplate",
				Name:       name,
				APIVersion: "v1beta3",
			},
		},
	}}

	rcOutData, err := yaml.Marshal(rc)
	checkErr(err)

	fmt.Print(string(rcOutData))
}

// TODO: what defaults make the most sense?
func portsFromString(spec string) (servicePort int, containerPort int, err error) {
	if spec == "" {
		return 0, 0, fmt.Errorf("empty port spec")
	}
	pieces := strings.Split(spec, ":")
	if len(pieces) != 2 {
		glog.Infof("Bad port spec: %s", spec)
		return 0, 0, fmt.Errorf("bad port spec: %s", spec)
	}
	servicePort, err = strconv.Atoi(pieces[0])
	if err != nil {
		glog.Errorf("Service port is not integer: %s %v", pieces[0], err)
		return 0, 0, err
	}
	if servicePort < 1 {
		glog.Errorf("Service port is not valid: %d", servicePort)
		return 0, 0, err
	}
	containerPort, err = strconv.Atoi(pieces[1])
	if err != nil {
		glog.Errorf("Container port is not integer: %s %v", pieces[1], err)
		return 0, 0, err
	}
	if containerPort < 1 {
		glog.Errorf("Container port is not valid: %d", containerPort)
		return 0, 0, err
	}

	return
}

//////////////////////////////////////////////////////////////////////

// Client tool utility functions copied from kubectl, kubecfg, and podex.
// This should probably be a separate package, but the right solution is
// to refactor the copied code and delete it from here.

func ReadConfigData(location string) ([]byte, error) {
	if len(location) == 0 {
		return nil, fmt.Errorf("location given but empty")
	}

	if location == "-" {
		// Read from stdin.
		data, err := ioutil.ReadAll(os.Stdin)
		if err != nil {
			return nil, err
		}

		if len(data) == 0 {
			return nil, fmt.Errorf(`Read from stdin specified ("-") but no data found`)
		}

		return data, nil
	}

	// Use the location as a file path or URL.
	return readConfigDataFromLocation(location)
}

func readConfigDataFromLocation(location string) ([]byte, error) {
	// we look for http:// or https:// to determine if valid URL, otherwise do normal file IO
	if strings.Index(location, "http://") == 0 || strings.Index(location, "https://") == 0 {
		resp, err := http.Get(location)
		if err != nil {
			return nil, fmt.Errorf("unable to access URL %s: %v\n", location, err)
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			return nil, fmt.Errorf("unable to read URL, server reported %d %s", resp.StatusCode, resp.Status)
		}
		data, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return nil, fmt.Errorf("unable to read URL %s: %v\n", location, err)
		}
		return data, nil
	} else {
		data, err := ioutil.ReadFile(location)
		if err != nil {
			return nil, fmt.Errorf("unable to read %s: %v\n", location, err)
		}
		return data, nil
	}
}

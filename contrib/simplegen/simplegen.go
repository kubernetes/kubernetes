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

// simplegen is a tool to generate simple services from a simple description
//
// $ simplegen myservice.json | kubectl create -f -
// $ simplegen myservice.yaml | kubectl create -f -
//
// This is completely separate from kubectl at the moment, until we figure out
// what the right integration approach is.

package main

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strconv"
	"strings"

	// TODO: handle multiple versions correctly
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/ghodss/yaml"
	"github.com/golang/glog"
)

// TODO: Also handle lists of simple services, and multiple input files

const usage = "usage: simplegen filename"

type SimpleService struct {
	// Optional: Defaults to image base name if not specified
	Name string `json:"name,omitempty"`
	// Required.
	Image string `json:"image"`
	// Optional: Defaults to one
	Replicas int `json:"replicas,omitempty"`
	// Optional: Creates a service if specified: servicePort:containerPort
	PortSpec string `json:"portSpec,omitempty"`
}

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

	simpleService := readSimpleService(filename)

	var servicePort, containerPort int
	var err error
	var ports []v1beta1.Port
	if simpleService.PortSpec != "" {
		servicePort, containerPort, err = portsFromString(simpleService.PortSpec)
		checkErr(err)

		generateService(simpleService.Name, servicePort, containerPort)

		// For replication controller
		ports = []v1beta1.Port{{Name: "main", ContainerPort: containerPort}}
	}

	generateReplicationController(simpleService.Name, simpleService.Image, simpleService.Replicas, ports)
}

func generateService(name string, servicePort int, containerPort int) {
	svc := []v1beta1.Service{{
		TypeMeta:      v1beta1.TypeMeta{APIVersion: "v1beta1", Kind: "Service", ID: name},
		Port:          servicePort,
		ContainerPort: util.NewIntOrStringFromInt(containerPort),
		Labels: map[string]string{
			"simpleservice": name,
		},
		Selector: map[string]string{
			"simpleservice": name,
		},
	}}

	svcOutData, err := yaml.Marshal(svc)
	checkErr(err)

	fmt.Print(string(svcOutData))
}

func generateReplicationController(name string, image string, replicas int, ports []v1beta1.Port) {
	controller := []v1beta1.ReplicationController{{
		TypeMeta: v1beta1.TypeMeta{APIVersion: "v1beta1", Kind: "ReplicationController", ID: name},
		DesiredState: v1beta1.ReplicationControllerState{
			Replicas: replicas,
			ReplicaSelector: map[string]string{
				"simpleservice": name,
			},
			PodTemplate: v1beta1.PodTemplate{
				DesiredState: v1beta1.PodState{
					Manifest: v1beta1.ContainerManifest{
						Version: "v1beta2",
						Containers: []v1beta1.Container{
							{
								Name:  name,
								Image: image,
								Ports: ports,
							},
						},
					},
				},
				Labels: map[string]string{
					"simpleservice": name,
				},
			},
		},
		Labels: map[string]string{
			"simpleservice": name,
		},
	}}
	controllerOutData, err := yaml.Marshal(controller)
	checkErr(err)

	fmt.Print(string(controllerOutData))
}

func readSimpleService(filename string) SimpleService {
	inData, err := ReadConfigData(filename)
	checkErr(err)

	simpleService := SimpleService{}
	err = yaml.Unmarshal(inData, &simpleService)
	checkErr(err)

	if simpleService.Name == "" {
		_, simpleService.Name = ParseDockerImage(simpleService.Image)
		// TODO: encode/scrub the name
	}
	simpleService.Name = strings.ToLower(simpleService.Name)

	// TODO: Validate the image name and extract exposed ports

	// TODO: Do more validation
	if !util.IsDNSLabel(simpleService.Name) {
		checkErr(fmt.Errorf("name (%s) is not a valid DNS label", simpleService.Name))
	}

	if simpleService.Replicas == 0 {
		simpleService.Replicas = 1
	}

	return simpleService
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

// ParseDockerImage split a docker image name of the form [REGISTRYHOST/][USERNAME/]NAME[:TAG]
// TODO: handle the TAG
// Returns array of images name parts and base image name
func ParseDockerImage(imageName string) (parts []string, baseName string) {
	// Parse docker image name
	// IMAGE: [REGISTRYHOST/][USERNAME/]NAME[:TAG]
	// NAME: [a-z0-9-_.]
	parts = strings.Split(imageName, "/")
	baseName = parts[len(parts)-1]
	return
}

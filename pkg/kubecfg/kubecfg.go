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

package kubecfg

import (
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	"github.com/golang/glog"
	"gopkg.in/v1/yaml"
)

func GetServerVersion(client *client.Client) (*version.Info, error) {
	info, err := client.ServerVersion()
	if err != nil {
		return nil, fmt.Errorf("Got error: %v", err)
	}
	return info, nil
}

func promptForString(field string, r io.Reader) string {
	fmt.Printf("Please enter %s: ", field)
	var result string
	fmt.Fscan(r, &result)
	return result
}

type AuthInfo struct {
	User     string
	Password string
	CAFile   string
	CertFile string
	KeyFile  string
	Insecure *bool
}

// LoadAuthInfo parses an AuthInfo object from a file path. It prompts user and creates file if it doesn't exist.
func LoadAuthInfo(path string, r io.Reader) (*AuthInfo, error) {
	var auth AuthInfo
	if _, err := os.Stat(path); os.IsNotExist(err) {
		auth.User = promptForString("Username", r)
		auth.Password = promptForString("Password", r)
		data, err := json.Marshal(auth)
		if err != nil {
			return &auth, err
		}
		err = ioutil.WriteFile(path, data, 0600)
		return &auth, err
	}
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(data, &auth)
	if err != nil {
		return nil, err
	}
	return &auth, err
}

// Update performs a rolling update of a collection of pods.
// 'name' points to a replication controller.
// 'client' is used for updating pods.
// 'updatePeriod' is the time between pod updates.
// 'imageName' is the new image to update for the template.  This will work
//     with the first container in the pod.  There is no support yet for
//     updating more complex replication controllers.  If this is blank then no
//     update of the image is performed.
func Update(ctx api.Context, name string, client client.Interface, updatePeriod time.Duration, imageName string) error {
	controller, err := client.GetReplicationController(ctx, name)
	if err != nil {
		return err
	}

	if len(imageName) != 0 {
		controller.DesiredState.PodTemplate.DesiredState.Manifest.Containers[0].Image = imageName
		controller, err = client.UpdateReplicationController(ctx, controller)
		if err != nil {
			return err
		}
	}

	s := labels.Set(controller.DesiredState.ReplicaSelector).AsSelector()

	podList, err := client.ListPods(ctx, s)
	if err != nil {
		return err
	}
	expected := len(podList.Items)
	if expected == 0 {
		return nil
	}
	for _, pod := range podList.Items {
		// We delete the pod here, the controller will recreate it.  This will result in pulling
		// a new Docker image.  This isn't a full "update" but it's what we support for now.
		err = client.DeletePod(ctx, pod.ID)
		if err != nil {
			return err
		}
		time.Sleep(updatePeriod)
	}
	return wait.Poll(time.Second*5, time.Second*300, func() (bool, error) {
		podList, err := client.ListPods(ctx, s)
		if err != nil {
			return false, err
		}
		return len(podList.Items) == expected, nil
	})
}

// StopController stops a controller named 'name' by setting replicas to zero.
func StopController(ctx api.Context, name string, client client.Interface) error {
	return ResizeController(ctx, name, 0, client)
}

// ResizeController resizes a controller named 'name' by setting replicas to 'replicas'.
func ResizeController(ctx api.Context, name string, replicas int, client client.Interface) error {
	controller, err := client.GetReplicationController(ctx, name)
	if err != nil {
		return err
	}
	controller.DesiredState.Replicas = replicas
	controllerOut, err := client.UpdateReplicationController(ctx, controller)
	if err != nil {
		return err
	}
	data, err := yaml.Marshal(controllerOut)
	if err != nil {
		return err
	}
	fmt.Print(string(data))
	return nil
}

func portsFromString(spec string) []api.Port {
	if spec == "" {
		return []api.Port{}
	}
	parts := strings.Split(spec, ",")
	var result []api.Port
	for _, part := range parts {
		pieces := strings.Split(part, ":")
		if len(pieces) < 1 || len(pieces) > 2 {
			glog.Infof("Bad port spec: %s", part)
			continue
		}
		host := 0
		container := 0
		var err error
		if len(pieces) == 1 {
			container, err = strconv.Atoi(pieces[0])
			if err != nil {
				glog.Errorf("Container port is not integer: %s %v", pieces[0], err)
				continue
			}
		} else {
			host, err = strconv.Atoi(pieces[0])
			if err != nil {
				glog.Errorf("Host port is not integer: %s %v", pieces[0], err)
				continue
			}
			container, err = strconv.Atoi(pieces[1])
			if err != nil {
				glog.Errorf("Container port is not integer: %s %v", pieces[1], err)
				continue
			}
		}
		if container < 1 {
			glog.Errorf("Container port is not valid: %d", container)
			continue
		}

		result = append(result, api.Port{ContainerPort: container, HostPort: host})
	}
	return result
}

// RunController creates a new replication controller named 'name' which creates 'replicas' pods running 'image'.
func RunController(ctx api.Context, image, name string, replicas int, client client.Interface, portSpec string, servicePort int) error {
	if servicePort > 0 && !util.IsDNSLabel(name) {
		return fmt.Errorf("Service creation requested, but an invalid name for a service was provided (%s). Service names must be valid DNS labels.", name)
	}
	controller := &api.ReplicationController{
		JSONBase: api.JSONBase{
			ID: name,
		},
		DesiredState: api.ReplicationControllerState{
			Replicas: replicas,
			ReplicaSelector: map[string]string{
				"simpleService": name,
			},
			PodTemplate: api.PodTemplate{
				DesiredState: api.PodState{
					Manifest: api.ContainerManifest{
						Version: "v1beta2",
						Containers: []api.Container{
							{
								Name:  strings.ToLower(name),
								Image: image,
								Ports: portsFromString(portSpec),
							},
						},
					},
				},
				Labels: map[string]string{
					"simpleService": name,
				},
			},
		},
	}

	controllerOut, err := client.CreateReplicationController(ctx, controller)
	if err != nil {
		return err
	}
	data, err := yaml.Marshal(controllerOut)
	if err != nil {
		return err
	}
	fmt.Print(string(data))

	if servicePort > 0 {
		svc, err := createService(ctx, name, servicePort, client)
		if err != nil {
			return err
		}
		data, err = yaml.Marshal(svc)
		if err != nil {
			return err
		}
		fmt.Printf(string(data))
	}
	return nil
}

func createService(ctx api.Context, name string, port int, client client.Interface) (*api.Service, error) {
	svc := &api.Service{
		JSONBase: api.JSONBase{ID: name},
		Port:     port,
		Labels: map[string]string{
			"simpleService": name,
		},
		Selector: map[string]string{
			"simpleService": name,
		},
	}
	svc, err := client.CreateService(ctx, svc)
	return svc, err
}

// DeleteController deletes a replication controller named 'name', requires that the controller
// already be stopped.
func DeleteController(ctx api.Context, name string, client client.Interface) error {
	controller, err := client.GetReplicationController(ctx, name)
	if err != nil {
		return err
	}
	if controller.DesiredState.Replicas != 0 {
		return fmt.Errorf("controller has non-zero replicas (%d), please stop it first", controller.DesiredState.Replicas)
	}
	return client.DeleteReplicationController(ctx, name)
}

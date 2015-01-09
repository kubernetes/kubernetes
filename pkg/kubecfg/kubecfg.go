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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/version"
	"github.com/ghodss/yaml"
	"github.com/golang/glog"
)

func GetServerVersion(client *client.Client) (*version.Info, error) {
	info, err := client.ServerVersion()
	if err != nil {
		return nil, err
	}
	return info, nil
}

func promptForString(field string, r io.Reader) string {
	fmt.Printf("Please enter %s: ", field)
	var result string
	fmt.Fscan(r, &result)
	return result
}

type NamespaceInfo struct {
	Namespace string
}

// LoadClientAuthInfoOrPrompt parses a clientauth.Info object from a file path. It prompts user and creates file if it doesn't exist.
// Oddly, it returns a clientauth.Info even if there is an error.
func LoadClientAuthInfoOrPrompt(path string, r io.Reader) (*clientauth.Info, error) {
	var auth clientauth.Info
	// Prompt for user/pass and write a file if none exists.
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
	authPtr, err := clientauth.LoadFromFile(path)
	if err != nil {
		return nil, err
	}
	return authPtr, nil
}

// LoadNamespaceInfo parses a NamespaceInfo object from a file path. It creates a file at the specified path if it doesn't exist with the default namespace.
func LoadNamespaceInfo(path string) (*NamespaceInfo, error) {
	var ns NamespaceInfo
	if _, err := os.Stat(path); os.IsNotExist(err) {
		ns.Namespace = api.NamespaceDefault
		err = SaveNamespaceInfo(path, &ns)
		return &ns, err
	}
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, err
	}
	err = json.Unmarshal(data, &ns)
	if err != nil {
		return nil, err
	}
	return &ns, err
}

// SaveNamespaceInfo saves a NamespaceInfo object at the specified file path.
func SaveNamespaceInfo(path string, ns *NamespaceInfo) error {
	if !util.IsDNSLabel(ns.Namespace) {
		return fmt.Errorf("namespace %s is not a valid DNS Label", ns.Namespace)
	}
	data, err := json.Marshal(ns)
	err = ioutil.WriteFile(path, data, 0600)
	return err
}

// extracted for test speed
var (
	updatePollInterval = 5 * time.Second
	updatePollTimeout  = 300 * time.Second
)

// Update performs a rolling update of a collection of pods.
// 'name' points to a replication controller.
// 'client' is used for updating pods.
// 'updatePeriod' is the time between pod updates.
// 'imageName' is the new image to update for the template.  This will work
//     with the first container in the pod.  There is no support yet for
//     updating more complex replication controllers.  If this is blank then no
//     update of the image is performed.
func Update(ctx api.Context, name string, client client.Interface, updatePeriod time.Duration, imageName string) error {
	// TODO ctx is not needed as input to this function, should just be 'namespace'
	controller, err := client.ReplicationControllers(api.Namespace(ctx)).Get(name)
	if err != nil {
		return err
	}

	if len(imageName) != 0 {
		controller.Spec.Template.Spec.Containers[0].Image = imageName
		controller, err = client.ReplicationControllers(controller.Namespace).Update(controller)
		if err != nil {
			return err
		}
	}

	s := labels.Set(controller.Spec.Selector).AsSelector()

	podList, err := client.Pods(api.Namespace(ctx)).List(s)
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
		err = client.Pods(pod.Namespace).Delete(pod.Name)
		if err != nil {
			return err
		}
		time.Sleep(updatePeriod)
	}
	return wait.Poll(updatePollInterval, updatePollTimeout, func() (bool, error) {
		podList, err := client.Pods(api.Namespace(ctx)).List(s)
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
	// TODO ctx is not needed, and should just be a namespace
	controller, err := client.ReplicationControllers(api.Namespace(ctx)).Get(name)
	if err != nil {
		return err
	}
	controller.Spec.Replicas = replicas
	controllerOut, err := client.ReplicationControllers(api.Namespace(ctx)).Update(controller)
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

func portsFromString(spec string) ([]api.Port, error) {
	if spec == "" {
		return []api.Port{}, nil
	}
	parts := strings.Split(spec, ",")
	var result []api.Port
	for _, part := range parts {
		pieces := strings.Split(part, ":")
		if len(pieces) < 1 || len(pieces) > 2 {
			glog.Infof("Bad port spec: %s", part)
			return nil, fmt.Errorf("bad port spec: %s", part)
		}
		host := 0
		container := 0
		var err error
		if len(pieces) == 1 {
			container, err = strconv.Atoi(pieces[0])
			if err != nil {
				glog.Errorf("Container port is not integer: %s %v", pieces[0], err)
				return nil, err
			}
		} else {
			host, err = strconv.Atoi(pieces[0])
			if err != nil {
				glog.Errorf("Host port is not integer: %s %v", pieces[0], err)
				return nil, err
			}
			container, err = strconv.Atoi(pieces[1])
			if err != nil {
				glog.Errorf("Container port is not integer: %s %v", pieces[1], err)
				return nil, err
			}
		}
		if container < 1 {
			glog.Errorf("Container port is not valid: %d", container)
			return nil, err
		}

		result = append(result, api.Port{ContainerPort: container, HostPort: host})
	}
	return result, nil
}

// RunController creates a new replication controller named 'name' which creates 'replicas' pods running 'image'.
func RunController(ctx api.Context, image, name string, replicas int, client client.Interface, portSpec string, servicePort int) error {
	// TODO replace ctx with a namespace string
	if servicePort > 0 && !util.IsDNSLabel(name) {
		return fmt.Errorf("service creation requested, but an invalid name for a service was provided (%s). Service names must be valid DNS labels.", name)
	}
	ports, err := portsFromString(portSpec)
	if err != nil {
		return err
	}
	controller := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: name,
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: replicas,
			Selector: map[string]string{
				"name": name,
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{
						"name": name,
					},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  strings.ToLower(name),
							Image: image,
							Ports: ports,
						},
					},
				},
			},
		},
	}

	controllerOut, err := client.ReplicationControllers(api.Namespace(ctx)).Create(controller)
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
	// TODO remove context in favor of just namespace string
	svc := &api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"name": name,
			},
		},
		Spec: api.ServiceSpec{
			Port: port,
			Selector: map[string]string{
				"name": name,
			},
		},
	}
	svc, err := client.Services(api.Namespace(ctx)).Create(svc)
	return svc, err
}

// DeleteController deletes a replication controller named 'name', requires that the controller
// already be stopped.
func DeleteController(ctx api.Context, name string, client client.Interface) error {
	// TODO remove ctx in favor of just namespace string
	controller, err := client.ReplicationControllers(api.Namespace(ctx)).Get(name)
	if err != nil {
		return err
	}
	if controller.Spec.Replicas != 0 {
		return fmt.Errorf("controller has non-zero replicas (%d), please stop it first", controller.Spec.Replicas)
	}
	return client.ReplicationControllers(api.Namespace(ctx)).Delete(name)
}

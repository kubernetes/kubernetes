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

package e2e

import (
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

func TestNetwork(c *client.Client) bool {
	if testContext.provider == "vagrant" {
		glog.Infof("Skipping test which is broken for vagrant (See https://github.com/GoogleCloudPlatform/kubernetes/issues/3580)")
		return true
	}

	ns := api.NamespaceDefault
	// TODO(satnam6502): Replace call of randomSuffix with call to NewUUID when service
	//                   names have the same form as pod and replication controller names.
	name := "nettest-" + randomSuffix()
	svc, err := c.Services(ns).Create(&api.Service{
		ObjectMeta: api.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"name": name,
			},
		},
		Spec: api.ServiceSpec{
			Port:          8080,
			ContainerPort: util.NewIntOrStringFromInt(8080),
			Selector: map[string]string{
				"name": name,
			},
		},
	})
	glog.Infof("Creating service with name %s", svc.Name)
	if err != nil {
		glog.Errorf("unable to create test service %s: %v", svc.Name, err)
		return false
	}
	// Clean up service
	defer func() {
		if err = c.Services(ns).Delete(svc.Name); err != nil {
			glog.Errorf("unable to delete svc %v: %v", svc.Name, err)
		}
	}()
	rc, err := c.ReplicationControllers(ns).Create(&api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name: name,
			Labels: map[string]string{
				"name": name,
			},
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 8,
			Selector: map[string]string{
				"name": name,
			},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"name": name},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:    "webserver",
							Image:   "kubernetes/nettest:latest",
							Command: []string{"-service=" + name},
							Ports:   []api.Port{{ContainerPort: 8080}},
						},
					},
				},
			},
		},
	})
	if err != nil {
		glog.Errorf("unable to create test rc: %v", err)
		return false
	}
	// Clean up rc
	defer func() {
		rc.Spec.Replicas = 0
		rc, err = c.ReplicationControllers(ns).Update(rc)
		if err != nil {
			glog.Errorf("unable to modify replica count for rc %v: %v", rc.Name, err)
			return
		}
		if err = c.ReplicationControllers(ns).Delete(rc.Name); err != nil {
			glog.Errorf("unable to delete rc %v: %v", rc.Name, err)
		}
	}()
	const maxAttempts = 60
	for i := 0; i < maxAttempts; i++ {
		time.Sleep(2 * time.Second)
		body, err := c.Get().Prefix("proxy").Resource("services").Name(svc.Name).Suffix("status").Do().Raw()
		if err != nil {
			glog.Infof("Attempt %v/%v: service/pod still starting. (error: '%v')", i, maxAttempts, err)
			continue
		}
		switch string(body) {
		case "pass":
			glog.Infof("Passed on attempt %v. Cleaning up.", i)
			return true
		case "running":
			glog.Infof("Attempt %v/%v: test still running", i, maxAttempts)
		case "fail":
			if body, err := c.Get().Prefix("proxy").Resource("services").Name(svc.Name).Suffix("read").Do().Raw(); err != nil {
				glog.Infof("Failed on attempt %v. Cleaning up. Error reading details: %v", i, err)
			} else {
				glog.Infof("Failed on attempt %v. Cleaning up. Details:\n%v", i, string(body))
			}
			return false
		}
	}

	if body, err := c.Get().Prefix("proxy").Resource("services").Name(svc.Name).Suffix("read").Do().Raw(); err != nil {
		glog.Infof("Timed out. Cleaning up. Error reading details: %v", err)
	} else {
		glog.Infof("Timed out. Cleaning up. Details:\n%v", string(body))
	}

	return false
}

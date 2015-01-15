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

package cmd_test

import (
	"fmt"
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl"
	. "github.com/GoogleCloudPlatform/kubernetes/pkg/kubectl/cmd"
)

func ExamplePrintReplicationController() {
	f, tf, codec := NewAPIFactory()
	tf.Printer = kubectl.NewHumanReadablePrinter(false)
	tf.Client = &client.FakeRESTClient{
		Codec:  codec,
		Client: nil,
	}
	cmd := f.NewCmdRunContainer(os.Stdout)
	ctrl := &api.ReplicationController{
		ObjectMeta: api.ObjectMeta{
			Name:   "foo",
			Labels: map[string]string{"foo": "bar"},
		},
		Spec: api.ReplicationControllerSpec{
			Replicas: 1,
			Selector: map[string]string{"foo": "bar"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: api.ObjectMeta{
					Labels: map[string]string{"foo": "bar"},
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{
							Name:  "foo",
							Image: "someimage",
						},
					},
				},
			},
		},
	}
	err := PrintObject(cmd, ctrl, f, os.Stdout)
	if err != nil {
		fmt.Printf("Unexpected error: %v", err)
	}
	// Output:
	// CONTROLLER          CONTAINER(S)        IMAGE(S)            SELECTOR            REPLICAS
	// foo                 foo                 someimage           foo=bar             1
}

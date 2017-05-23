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

package example

import (
	"fmt"

    "github.com/GoogleCloudPlatform/kubernetes/etcdex"
    )

edex = etcdex.NewClient()

services = edex.IndexKeys("/registry/service")
pod_labels = edex.IndexKeysAndValues("/registry/pods", "Labels")
service_ports = edex.InvertedIndex("/registry/service", "Port")

func CreateService(serviceToCreate) {
  if services.contains(servicetocreate.id) {
    fmt.Println("internal error: duplicate service id.")
   }

  service_id, err = service_ports.Get(servicetocreate.Port)
  if err != nil {
    fmt.Println("Warning: Service %s is already using port %d", service_id, serviceToCreate.Port)
   }

  pods_involved := 0
  // TODO: think about how I am going to iterate over a list of keys from an index that is changing underneath.
  for label in pod_labels {
    if Match(pod_labels, serviceToCreate.Selector) {
       pods_involved += 1
    }
  }
  fmt.Println("Info: %d pod match selector for new service %s", pods_involved, serviceToCreate.Id)
}


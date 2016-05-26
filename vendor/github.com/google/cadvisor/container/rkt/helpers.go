// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package rkt

import (
	"fmt"
	"io/ioutil"
	"path"
	"strings"

	rktapi "github.com/coreos/rkt/api/v1alpha"
	"github.com/golang/glog"
	"golang.org/x/net/context"
)

type parsedName struct {
	Pod       string
	Container string
}

func verifyPod(name string) (bool, error) {
	pod, err := cgroupToPod(name)

	if err != nil || pod == nil {
		return false, err
	}

	// Anything handler can handle is also accepted.
	// Accept cgroups that are sub the pod cgroup, except "system.slice"
	//   - "system.slice" doesn't contain any processes itself
	accept := !strings.HasSuffix(name, "/system.slice")

	return accept, nil
}

func cgroupToPod(name string) (*rktapi.Pod, error) {
	rktClient, err := Client()
	if err != nil {
		return nil, fmt.Errorf("couldn't get rkt api service: %v", err)
	}

	resp, err := rktClient.ListPods(context.Background(), &rktapi.ListPodsRequest{
		Filters: []*rktapi.PodFilter{
			{
				States:        []rktapi.PodState{rktapi.PodState_POD_STATE_RUNNING},
				PodSubCgroups: []string{name},
			},
		},
	})

	if err != nil {
		return nil, fmt.Errorf("failed to list pods: %v", err)
	}

	if len(resp.Pods) == 0 {
		return nil, nil
	}

	if len(resp.Pods) != 1 {
		return nil, fmt.Errorf("returned %d (expected 1) pods for cgroup %v", len(resp.Pods), name)
	}

	return resp.Pods[0], nil
}

/* Parse cgroup name into a pod/container name struct
   Example cgroup fs name

   pod - /machine.slice/machine-rkt\\x2df556b64a\\x2d17a7\\x2d47d7\\x2d93ec\\x2def2275c3d67e.scope/
    or   /system.slice/k8s-..../
   container under pod - /machine.slice/machine-rkt\\x2df556b64a\\x2d17a7\\x2d47d7\\x2d93ec\\x2def2275c3d67e.scope/system.slice/alpine-sh.service
    or  /system.slice/k8s-..../system.slice/pause.service
*/
func parseName(name string) (*parsedName, error) {
	pod, err := cgroupToPod(name)
	if err != nil {
		return nil, fmt.Errorf("parseName: couldn't convert %v to a rkt pod: %v", name, err)
	}
	if pod == nil {
		return nil, fmt.Errorf("parseName: didn't return a pod for %v", name)
	}

	splits := strings.Split(name, "/")

	parsed := &parsedName{}

	if len(splits) == 3 || len(splits) == 5 {
		parsed.Pod = pod.Id

		if len(splits) == 5 {
			parsed.Container = strings.Replace(splits[4], ".service", "", -1)
		}

		return parsed, nil
	}

	return nil, fmt.Errorf("%s not handled by rkt handler", name)
}

// Gets a Rkt container's overlay upper dir
func getRootFs(root string, parsed *parsedName) string {
	/* Example of where it stores the upper dir key
	for container
		/var/lib/rkt/pods/run/bc793ec6-c48f-4480-99b5-6bec16d52210/appsinfo/alpine-sh/treeStoreID
	for pod
		/var/lib/rkt/pods/run/f556b64a-17a7-47d7-93ec-ef2275c3d67e/stage1TreeStoreID

	*/

	var tree string
	if parsed.Container == "" {
		tree = path.Join(root, "pods/run", parsed.Pod, "stage1TreeStoreID")
	} else {
		tree = path.Join(root, "pods/run", parsed.Pod, "appsinfo", parsed.Container, "treeStoreID")
	}

	bytes, err := ioutil.ReadFile(tree)
	if err != nil {
		glog.Errorf("ReadFile failed, couldn't read %v to get upper dir: %v", tree, err)
		return ""
	}

	s := string(bytes)

	/* Example of where the upper dir is stored via key read above
	   /var/lib/rkt/pods/run/bc793ec6-c48f-4480-99b5-6bec16d52210/overlay/deps-sha512-82a099e560a596662b15dec835e9adabab539cad1f41776a30195a01a8f2f22b/
	*/
	return path.Join(root, "pods/run", parsed.Pod, "overlay", s)
}

/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package rkt

import (
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"google.golang.org/grpc"

	appcschema "github.com/appc/spec/schema"
	rktapi "github.com/coreos/rkt/api/v1alpha"
	"github.com/golang/glog"
	"golang.org/x/net/context"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
)

type runtimePodGetter struct {
	config *Config
	// The grpc client for rkt api-service.
	apisvcConn *grpc.ClientConn
	apisvc     rktapi.PublicAPIClient
}

func NewRuntimePodGetter(apiEndpoint string, config *Config) (kubecontainer.PodGetter, error) {
	apisvcConn, apisvc, config, err := getAPISvcAndConfig(apiEndpoint, config)
	if err != nil {
		return nil, err
	}

	return &runtimePodGetter{config: config, apisvcConn: apisvcConn, apisvc: apisvc}, nil
}

// GetPods runs 'systemctl list-unit' and 'rkt list' to get the list of rkt pods.
// Then it will use the result to construct a list of container runtime pods.
// If all is false, then only running pods will be returned, otherwise all pods will be
// returned.
func (rpg *runtimePodGetter) GetPods(all bool) ([]*kubecontainer.Pod, error) {
	glog.V(4).Infof("Rkt getting pods")

	listReq := &rktapi.ListPodsRequest{
		Detail: true,
		Filters: []*rktapi.PodFilter{
			{
				Annotations: []*rktapi.KeyValue{
					{
						Key:   k8sRktKubeletAnno,
						Value: k8sRktKubeletAnnoValue,
					},
				},
			},
		},
	}
	if !all {
		listReq.Filters[0].States = []rktapi.PodState{rktapi.PodState_POD_STATE_RUNNING}
	}
	listResp, err := rpg.apisvc.ListPods(context.Background(), listReq)
	if err != nil {
		return nil, fmt.Errorf("couldn't list pods: %v", err)
	}

	pods := make(map[types.UID]*kubecontainer.Pod)
	var podIDs []types.UID
	for _, pod := range listResp.Pods {
		pod, err := convertRktPod(pod)
		if err != nil {
			glog.Warningf("rkt: Cannot construct pod from unit file: %v.", err)
			continue
		}

		// Group pods together.
		oldPod, found := pods[pod.ID]
		if !found {
			pods[pod.ID] = pod
			podIDs = append(podIDs, pod.ID)
			continue
		}

		oldPod.Containers = append(oldPod.Containers, pod.Containers...)
	}

	// Convert map to list, using the consistent order from the podIDs array.
	var result []*kubecontainer.Pod
	for _, id := range podIDs {
		result = append(result, pods[id])
	}

	return result, nil
}

// convertRktPod will convert a rktapi.Pod to a kubecontainer.Pod
func convertRktPod(rktpod *rktapi.Pod) (*kubecontainer.Pod, error) {
	manifest := &appcschema.PodManifest{}
	err := json.Unmarshal(rktpod.Manifest, manifest)
	if err != nil {
		return nil, err
	}

	podUID, ok := manifest.Annotations.Get(k8sRktUIDAnno)
	if !ok {
		return nil, fmt.Errorf("pod is missing annotation %s", k8sRktUIDAnno)
	}
	podName, ok := manifest.Annotations.Get(k8sRktNameAnno)
	if !ok {
		return nil, fmt.Errorf("pod is missing annotation %s", k8sRktNameAnno)
	}
	podNamespace, ok := manifest.Annotations.Get(k8sRktNamespaceAnno)
	if !ok {
		return nil, fmt.Errorf("pod is missing annotation %s", k8sRktNamespaceAnno)
	}

	kubepod := &kubecontainer.Pod{
		ID:        types.UID(podUID),
		Name:      podName,
		Namespace: podNamespace,
	}

	for i, app := range rktpod.Apps {
		// The order of the apps is determined by the rkt pod manifest.
		// TODO(yifan): Let the server to unmarshal the annotations? https://github.com/coreos/rkt/issues/1872
		hashStr, ok := manifest.Apps[i].Annotations.Get(k8sRktContainerHashAnno)
		if !ok {
			return nil, fmt.Errorf("app %q is missing annotation %s", app.Name, k8sRktContainerHashAnno)
		}
		containerHash, err := strconv.ParseUint(hashStr, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("couldn't parse container's hash %q: %v", hashStr, err)
		}

		kubepod.Containers = append(kubepod.Containers, &kubecontainer.Container{
			ID:   buildContainerID(&containerID{rktpod.Id, app.Name}),
			Name: app.Name,
			// By default, the version returned by rkt API service will be "latest" if not specified.
			Image:   fmt.Sprintf("%s:%s", app.Image.Name, app.Image.Version),
			Hash:    containerHash,
			State:   appStateToContainerState(app.State),
			Created: time.Unix(0, rktpod.CreatedAt).Unix(), // convert ns to s
		})
	}

	return kubepod, nil
}

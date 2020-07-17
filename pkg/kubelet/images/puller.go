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

package images

import (
	"k8s.io/kubernetes/pkg/apis/scheduling"
	"k8s.io/kubernetes/pkg/util/env"
	"sort"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1alpha2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type pullResult struct {
	imageRef string
	err      error
}

type imagePuller interface {
	pullImage(kubecontainer.ImageSpec, []v1.Secret, chan<- pullResult, *runtimeapi.PodSandboxConfig, *v1.Pod)
}

var _, _ imagePuller = &parallelImagePuller{}, &serialImagePuller{}

type parallelImagePuller struct {
	imageService kubecontainer.ImageService
}

func newParallelImagePuller(imageService kubecontainer.ImageService) imagePuller {
	return &parallelImagePuller{imageService}
}

func (pip *parallelImagePuller) pullImage(spec kubecontainer.ImageSpec, pullSecrets []v1.Secret, pullChan chan<- pullResult, podSandboxConfig *runtimeapi.PodSandboxConfig, pod *v1.Pod) {
	go func() {
		imageRef, err := pip.imageService.PullImage(spec, pullSecrets, podSandboxConfig)
		pullChan <- pullResult{
			imageRef: imageRef,
			err:      err,
		}
	}()
}

// Maximum number of image pull requests than can be queued.
const maxImagePullRequests = 10

type serialImagePuller struct {
	imageService kubecontainer.ImageService
	pullRequests chan *imagePullRequest
}

func newSerialImagePuller(imageService kubecontainer.ImageService) imagePuller {
	imagePuller := &serialImagePuller{imageService, make(chan *imagePullRequest, maxImagePullRequests)}
	go wait.Until(imagePuller.processImagePullRequests, time.Second, wait.NeverStop)
	return imagePuller
}

type imagePullRequest struct {
	spec             kubecontainer.ImageSpec
	pod              *v1.Pod
	pullSecrets      []v1.Secret
	pullChan         chan<- pullResult
	podSandboxConfig *runtimeapi.PodSandboxConfig
}

func (sip *serialImagePuller) pullImage(spec kubecontainer.ImageSpec, pullSecrets []v1.Secret, pullChan chan<- pullResult, podSandboxConfig *runtimeapi.PodSandboxConfig, pod *v1.Pod) {
	sip.pullRequests <- &imagePullRequest{
		spec:             spec,
		pod:              pod,
		pullSecrets:      pullSecrets,
		pullChan:         pullChan,
		podSandboxConfig: podSandboxConfig,
	}
}

func (sip *serialImagePuller) processImagePullRequests() {
	// Set environment variables to identify the namespace that needs to be pulled first
	priorityNamespace := env.GetEnvAsStringOrFallback("PRIORITY_NAMESPACES", "kube-system")
	for request := range sip.pullRequests {

		// Wait for a while for as many requests as possible to enter the channel, Avoid first-come request
		// not being sorted and processed directly
		time.Sleep(1 * time.Second)
		pullRequestQ := make([]*imagePullRequest, 0)
		pullRequestQ = append(pullRequestQ, request)

		// Use a loop to fetch the requests in the channel at the current time and
		// store them in a slice to wait for sorting
	Loop:
		for {
			select {
			case request := <-sip.pullRequests:
				pullRequestQ = append(pullRequestQ, request)
			default:
				break Loop
			}
		}

		// Prioritize requests in the slice by PriorityClassName and Namespace
		sort.SliceStable(pullRequestQ, func(i, j int) bool {
			if pullRequestQ[i].pod.Spec.PriorityClassName == scheduling.SystemNodeCritical &&
				pullRequestQ[j].pod.Spec.PriorityClassName != scheduling.SystemNodeCritical {
				return true
			} else if pullRequestQ[i].pod.Spec.PriorityClassName != scheduling.SystemNodeCritical &&
				pullRequestQ[j].pod.Spec.PriorityClassName == scheduling.SystemNodeCritical {
				return false
			}
			if pullRequestQ[i].pod.Spec.PriorityClassName == scheduling.SystemClusterCritical &&
				pullRequestQ[j].pod.Spec.PriorityClassName != scheduling.SystemClusterCritical {
				return true
			} else if pullRequestQ[i].pod.Spec.PriorityClassName != scheduling.SystemClusterCritical &&
				pullRequestQ[j].pod.Spec.PriorityClassName == scheduling.SystemClusterCritical {
				return false
			}
			if strings.Contains(priorityNamespace, pullRequestQ[i].pod.Namespace) &&
				!strings.Contains(priorityNamespace, pullRequestQ[j].pod.Namespace) {
				return true
			} else if !strings.Contains(priorityNamespace, pullRequestQ[i].pod.Namespace) &&
				strings.Contains(priorityNamespace, pullRequestQ[j].pod.Namespace) {
				return false
			}
			return false
		})

		// Handle the requests one by one
		for _, pr := range pullRequestQ {
			imageRef, err := sip.imageService.PullImage(pr.spec, pr.pullSecrets, pr.podSandboxConfig)
			pr.pullChan <- pullResult{
				imageRef: imageRef,
				err:      err,
			}
		}
	}
}

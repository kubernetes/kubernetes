/*
Copyright 2017 The Kubernetes Authors.

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

package dockershim

import (
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	dockertypes "github.com/docker/docker/api/types"
	dockerfilters "github.com/docker/docker/api/types/filters"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	"k8s.io/kubernetes/pkg/kubelet/leaky"
)

// These are labels used by kuberuntime. Ideally, we should not rely on kuberuntime implementation
// detail in dockershim. However, we need these labels for legacy container (containers created by
// kubernetes 1.4 and 1.5) support.
// TODO(random-liu): Remove this file and related code in kubernetes 1.8.
const (
	podDeletionGracePeriodLabel    = "io.kubernetes.pod.deletionGracePeriod"
	podTerminationGracePeriodLabel = "io.kubernetes.pod.terminationGracePeriod"

	containerHashLabel                     = "io.kubernetes.container.hash"
	containerRestartCountLabel             = "io.kubernetes.container.restartCount"
	containerTerminationMessagePathLabel   = "io.kubernetes.container.terminationMessagePath"
	containerTerminationMessagePolicyLabel = "io.kubernetes.container.terminationMessagePolicy"
	containerPreStopHandlerLabel           = "io.kubernetes.container.preStopHandler"
	containerPortsLabel                    = "io.kubernetes.container.ports"
)

// NOTE that we can't handle the following dockershim internal labels, so they will be empty:
// * containerLogPathLabelKey ("io.kubernetes.container.logpath"): RemoveContainer will ignore
// the label if it's empty.
// * sandboxIDLabelKey ("io.kubernetes.sandbox.id"): This is used in 2 places:
//   * filter.PodSandboxId: The filter is not used in kuberuntime now.
//   * runtimeapi.Container.PodSandboxId: toRuntimeAPIContainer retrieves PodSandboxId from the
//   label. The field is used in kuberuntime sandbox garbage collection. Missing this may cause
//   pod sandbox to be removed before its containers are removed.

// convertLegacyNameAndLabels converts legacy name and labels into dockershim name and labels.
// The function can be used to either legacy infra container or regular container.
// NOTE that legacy infra container doesn't have restart count label, so the returned attempt for
// sandbox will always be 0. The sandbox attempt is only used to generate new sandbox name, and
// there is no naming conflict between legacy and new containers/sandboxes, so it should be fine.
func convertLegacyNameAndLabels(names []string, labels map[string]string) ([]string, map[string]string, error) {
	if len(names) == 0 {
		return nil, nil, fmt.Errorf("unexpected empty name")
	}

	// Generate new dockershim name.
	m, _, err := libdocker.ParseDockerName(names[0])
	if err != nil {
		return nil, nil, err
	}
	sandboxName, sandboxNamespace, err := kubecontainer.ParsePodFullName(m.PodFullName)
	if err != nil {
		return nil, nil, err
	}
	newNames := []string{strings.Join([]string{
		kubePrefix,                         // 0
		m.ContainerName,                    // 1: container name
		sandboxName,                        // 2: sandbox name
		sandboxNamespace,                   // 3: sandbox namesapce
		string(m.PodUID),                   // 4: sandbox uid
		labels[containerRestartCountLabel], // 5
	}, nameDelimiter)}

	// Generate new labels.
	legacyAnnotations := sets.NewString(
		containerHashLabel,
		containerRestartCountLabel,
		containerTerminationMessagePathLabel,
		containerTerminationMessagePolicyLabel,
		containerPreStopHandlerLabel,
		containerPortsLabel,
	)
	newLabels := map[string]string{}
	for k, v := range labels {
		if legacyAnnotations.Has(k) {
			// Add annotation prefix for all legacy labels which should be annotations in dockershim.
			newLabels[fmt.Sprintf("%s%s", annotationPrefix, k)] = v
		} else {
			newLabels[k] = v
		}
	}
	// Add containerTypeLabelKey indicating the container is sandbox or application container.
	if m.ContainerName == leaky.PodInfraContainerName {
		newLabels[containerTypeLabelKey] = containerTypeLabelSandbox
	} else {
		newLabels[containerTypeLabelKey] = containerTypeLabelContainer
	}
	return newNames, newLabels, nil
}

// legacyCleanupCheckInterval is the interval legacyCleanupCheck is performed.
const legacyCleanupCheckInterval = 10 * time.Second

// LegacyCleanupInit initializes the legacy cleanup flag. If necessary, it will starts a goroutine
// which periodically checks legacy cleanup until it finishes.
func (ds *dockerService) LegacyCleanupInit() {
	// If there is no legacy container/sandbox, just return.
	if clean, _ := ds.checkLegacyCleanup(); clean {
		return
	}
	// Or else start the cleanup routine.
	go wait.PollInfinite(legacyCleanupCheckInterval, ds.checkLegacyCleanup)
}

// checkLegacyCleanup lists legacy containers/sandboxes, if no legacy containers/sandboxes are found,
// mark the legacy cleanup flag as done.
func (ds *dockerService) checkLegacyCleanup() (bool, error) {
	// Always do legacy cleanup when list fails.
	sandboxes, err := ds.ListLegacyPodSandbox(nil)
	if err != nil {
		glog.Errorf("Failed to list legacy pod sandboxes: %v", err)
		return false, nil
	}
	containers, err := ds.ListLegacyContainers(nil)
	if err != nil {
		glog.Errorf("Failed to list legacy containers: %v", err)
		return false, nil
	}
	if len(sandboxes) != 0 || len(containers) != 0 {
		glog.V(4).Infof("Found legacy sandboxes %+v, legacy containers %+v, continue legacy cleanup",
			sandboxes, containers)
		return false, nil
	}
	ds.legacyCleanup.MarkDone()
	glog.V(2).Infof("No legacy containers found, stop performing legacy cleanup.")
	return true, nil
}

// ListLegacyPodSandbox only lists all legacy pod sandboxes.
func (ds *dockerService) ListLegacyPodSandbox(filter *runtimeapi.PodSandboxFilter) ([]*runtimeapi.PodSandbox, error) {
	// By default, list all containers whether they are running or not.
	opts := dockertypes.ContainerListOptions{All: true, Filters: dockerfilters.NewArgs()}
	filterOutReadySandboxes := false
	f := newDockerFilter(&opts.Filters)
	if filter != nil {
		if filter.Id != "" {
			f.Add("id", filter.Id)
		}
		if filter.State != nil {
			if filter.GetState().State == runtimeapi.PodSandboxState_SANDBOX_READY {
				// Only list running containers.
				opts.All = false
			} else {
				// runtimeapi.PodSandboxState_SANDBOX_NOTREADY can mean the
				// container is in any of the non-running state (e.g., created,
				// exited). We can't tell docker to filter out running
				// containers directly, so we'll need to filter them out
				// ourselves after getting the results.
				filterOutReadySandboxes = true
			}
		}
		if filter.LabelSelector != nil {
			for k, v := range filter.LabelSelector {
				f.AddLabel(k, v)
			}
		}
	}
	containers, err := ds.client.ListContainers(opts)
	if err != nil {
		return nil, err
	}

	// Convert docker containers to runtime api sandboxes.
	result := make([]*runtimeapi.PodSandbox, 0, len(containers))
	for i := range containers {
		c := containers[i]
		// Skip new containers with containerTypeLabelKey label.
		if _, ok := c.Labels[containerTypeLabelKey]; ok {
			continue
		}
		// If the container has no containerTypeLabelKey label, treat it as a legacy container.
		c.Names, c.Labels, err = convertLegacyNameAndLabels(c.Names, c.Labels)
		if err != nil {
			glog.V(4).Infof("Unable to convert legacy container %+v: %v", c, err)
			continue
		}
		if c.Labels[containerTypeLabelKey] != containerTypeLabelSandbox {
			continue
		}
		converted, err := containerToRuntimeAPISandbox(&c)
		if err != nil {
			glog.V(4).Infof("Unable to convert docker to runtime API sandbox %+v: %v", c, err)
			continue
		}
		if filterOutReadySandboxes && converted.State == runtimeapi.PodSandboxState_SANDBOX_READY {
			continue
		}
		result = append(result, converted)
	}
	return result, nil
}

// ListLegacyPodSandbox only lists all legacy containers.
func (ds *dockerService) ListLegacyContainers(filter *runtimeapi.ContainerFilter) ([]*runtimeapi.Container, error) {
	opts := dockertypes.ContainerListOptions{All: true, Filters: dockerfilters.NewArgs()}
	f := newDockerFilter(&opts.Filters)

	if filter != nil {
		if filter.Id != "" {
			f.Add("id", filter.Id)
		}
		if filter.State != nil {
			f.Add("status", toDockerContainerStatus(filter.GetState().State))
		}
		// NOTE: Legacy container doesn't have sandboxIDLabelKey label, so we can't filter with
		// it. Fortunately, PodSandboxId is not used in kuberuntime now.
		if filter.LabelSelector != nil {
			for k, v := range filter.LabelSelector {
				f.AddLabel(k, v)
			}
		}
	}
	containers, err := ds.client.ListContainers(opts)
	if err != nil {
		return nil, err
	}

	// Convert docker to runtime api containers.
	result := make([]*runtimeapi.Container, 0, len(containers))
	for i := range containers {
		c := containers[i]
		// Skip new containers with containerTypeLabelKey label.
		if _, ok := c.Labels[containerTypeLabelKey]; ok {
			continue
		}
		// If the container has no containerTypeLabelKey label, treat it as a legacy container.
		c.Names, c.Labels, err = convertLegacyNameAndLabels(c.Names, c.Labels)
		if err != nil {
			glog.V(4).Infof("Unable to convert legacy container %+v: %v", c, err)
			continue
		}
		if c.Labels[containerTypeLabelKey] != containerTypeLabelContainer {
			continue
		}
		converted, err := toRuntimeAPIContainer(&c)
		if err != nil {
			glog.V(4).Infof("Unable to convert docker container to runtime API container %+v: %v", c, err)
			continue
		}
		result = append(result, converted)
	}
	return result, nil
}

// legacyCleanupFlag is the flag indicating whether legacy cleanup is done.
type legacyCleanupFlag struct {
	done int32
}

// Done checks whether legacy cleanup is done.
func (f *legacyCleanupFlag) Done() bool {
	return atomic.LoadInt32(&f.done) == 1
}

// MarkDone sets legacy cleanup as done.
func (f *legacyCleanupFlag) MarkDone() {
	atomic.StoreInt32(&f.done, 1)
}

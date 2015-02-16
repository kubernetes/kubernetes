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

package kubelet

import (
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	execprobe "github.com/GoogleCloudPlatform/kubernetes/pkg/probe/exec"
	httprobe "github.com/GoogleCloudPlatform/kubernetes/pkg/probe/http"
	tcprobe "github.com/GoogleCloudPlatform/kubernetes/pkg/probe/tcp"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"

	"github.com/fsouza/go-dockerclient"
	"github.com/golang/glog"
)

const maxProbeRetries = 3

// probeContainer executes the given probe on a container and returns the result.
// If the probe is nil this returns Success. If the probe's initial delay has not passed
// since the creation of the container, this returns the defaultResult. It will then attempt
// to execute the probe repeatedly up to maxProbeRetries times, and return on the first
// successful result, else returning the last unsucessful result and error.
func (kl *Kubelet) probeContainer(p *api.Probe,
	podFullName string,
	podUID types.UID,
	status api.PodStatus,
	container api.Container,
	dockerContainer *docker.APIContainers,
	defaultResult probe.Result) (probe.Result, error) {
	var err error
	result := probe.Unknown
	if p == nil {
		return probe.Success, nil
	}
	if time.Now().Unix()-dockerContainer.Created < p.InitialDelaySeconds {
		return defaultResult, nil
	}
	for i := 0; i < maxProbeRetries; i++ {
		result, err = kl.runProbe(p, podFullName, podUID, status, container)
		if result == probe.Success {
			return result, err
		}
	}
	return result, err
}

func (kl *Kubelet) runProbe(p *api.Probe, podFullName string, podUID types.UID, status api.PodStatus, container api.Container) (probe.Result, error) {
	timeout := time.Duration(p.TimeoutSeconds) * time.Second
	if p.Exec != nil {
		return kl.prober.exec.Probe(kl.newExecInContainer(podFullName, podUID, container))
	}
	if p.HTTPGet != nil {
		port, err := extractPort(p.HTTPGet.Port, container)
		if err != nil {
			return probe.Unknown, err
		}
		host, port, path := extractGetParams(p.HTTPGet, status, port)
		return kl.prober.http.Probe(host, port, path, timeout)
	}
	if p.TCPSocket != nil {
		port, err := extractPort(p.TCPSocket.Port, container)
		if err != nil {
			return probe.Unknown, err
		}
		return kl.prober.tcp.Probe(status.PodIP, port, timeout)
	}
	glog.Warningf("Failed to find probe builder for %s %+v", container.Name, container.LivenessProbe)
	return probe.Unknown, nil
}

func extractGetParams(action *api.HTTPGetAction, status api.PodStatus, port int) (string, int, string) {
	host := action.Host
	if host == "" {
		host = status.PodIP
	}
	return host, port, action.Path
}

func extractPort(param util.IntOrString, container api.Container) (int, error) {
	port := -1
	var err error
	switch param.Kind {
	case util.IntstrInt:
		port := param.IntVal
		if port > 0 && port < 65536 {
			return port, nil
		}
		return port, fmt.Errorf("invalid port number: %v", port)
	case util.IntstrString:
		port = findPortByName(container, param.StrVal)
		if port == -1 {
			// Last ditch effort - maybe it was an int stored as string?
			if port, err = strconv.Atoi(param.StrVal); err != nil {
				return port, err
			}
		}
		if port > 0 && port < 65536 {
			return port, nil
		}
		return port, fmt.Errorf("invalid port number: %v", port)
	default:
		return port, fmt.Errorf("IntOrString had no kind: %+v", param)
	}
}

// findPortByName is a helper function to look up a port in a container by name.
// Returns the HostPort if found, -1 if not found.
func findPortByName(container api.Container, portName string) int {
	for _, port := range container.Ports {
		if port.Name == portName {
			return port.HostPort
		}
	}
	return -1
}

type execInContainer struct {
	run func() ([]byte, error)
}

func (kl *Kubelet) newExecInContainer(podFullName string, podUID types.UID, container api.Container) exec.Cmd {
	return execInContainer{func() ([]byte, error) {
		return kl.RunInContainer(podFullName, podUID, container.Name, container.LivenessProbe.Exec.Command)
	}}
}

func (eic execInContainer) CombinedOutput() ([]byte, error) {
	return eic.run()
}

func (eic execInContainer) SetDir(dir string) {
	//unimplemented
}

// This will eventually maintain info about probe results over time
// to allow for implementation of health thresholds
func newReadinessStates() *readinessStates {
	return &readinessStates{states: make(map[string]bool)}
}

type readinessStates struct {
	sync.Mutex
	states map[string]bool
}

func (r *readinessStates) IsReady(c api.ContainerStatus) bool {
	if c.State.Running == nil {
		return false
	}
	return r.get(strings.TrimPrefix(c.ContainerID, "docker://"))
}

func (r *readinessStates) get(key string) bool {
	r.Lock()
	defer r.Unlock()
	state, found := r.states[key]
	return state && found
}

func (r *readinessStates) set(key string, value bool) {
	r.Lock()
	defer r.Unlock()
	r.states[key] = value
}

func (r *readinessStates) remove(key string) {
	r.Lock()
	defer r.Unlock()
	delete(r.states, key)
}

func newProbeHolder() probeHolder {
	return probeHolder{
		exec: execprobe.New(),
		http: httprobe.New(),
		tcp:  tcprobe.New(),
	}
}

type probeHolder struct {
	exec execprobe.ExecProber
	http httprobe.HTTPProber
	tcp  tcprobe.TCPProber
}

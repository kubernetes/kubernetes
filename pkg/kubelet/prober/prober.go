/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package prober

import (
	"fmt"
	"io"
	"strconv"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/record"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/probe"
	execprobe "github.com/GoogleCloudPlatform/kubernetes/pkg/probe/exec"
	httprobe "github.com/GoogleCloudPlatform/kubernetes/pkg/probe/http"
	tcprobe "github.com/GoogleCloudPlatform/kubernetes/pkg/probe/tcp"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"

	"github.com/golang/glog"
)

const maxProbeRetries = 3

// Prober checks the healthiness of a container.
type Prober interface {
	Probe(pod *api.Pod, status api.PodStatus, container api.Container, containerID string, createdAt int64) (probe.Result, error)
}

type ContainerCommandRunner interface {
	RunInContainer(containerID string, cmd []string) ([]byte, error)
	ExecInContainer(containerID string, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool) error
	PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error
}

// Prober helps to check the liveness/readiness of a container.
type prober struct {
	exec execprobe.ExecProber
	http httprobe.HTTPProber
	tcp  tcprobe.TCPProber
	// TODO(vmarmol): Remove when we remove the circular dependency to DockerManager.
	Runner ContainerCommandRunner

	readinessManager *kubecontainer.ReadinessManager
	refManager       *kubecontainer.RefManager
	recorder         record.EventRecorder
}

// NewProber creates a Prober, it takes a command runner and
// several container info managers.
func New(
	runner ContainerCommandRunner,
	readinessManager *kubecontainer.ReadinessManager,
	refManager *kubecontainer.RefManager,
	recorder record.EventRecorder) Prober {

	return &prober{
		exec:   execprobe.New(),
		http:   httprobe.New(),
		tcp:    tcprobe.New(),
		Runner: runner,

		readinessManager: readinessManager,
		refManager:       refManager,
		recorder:         recorder,
	}
}

// New prober for use in tests.
func NewTestProber(
	exec execprobe.ExecProber,
	readinessManager *kubecontainer.ReadinessManager,
	refManager *kubecontainer.RefManager,
	recorder record.EventRecorder) Prober {

	return &prober{
		exec:             exec,
		readinessManager: readinessManager,
		refManager:       refManager,
		recorder:         recorder,
	}
}

// Probe checks the liveness/readiness of the given container.
// If the container's liveness probe is unsuccessful, set readiness to false.
// If liveness is successful, do a readiness check and set readiness accordingly.
func (pb *prober) Probe(pod *api.Pod, status api.PodStatus, container api.Container, containerID string, createdAt int64) (probe.Result, error) {
	// Probe liveness.
	live, err := pb.probeLiveness(pod, status, container, containerID, createdAt)
	if err != nil {
		glog.V(1).Infof("Liveness probe errored: %v", err)
		pb.readinessManager.SetReadiness(containerID, false)
		return probe.Unknown, err
	}
	if live != probe.Success {
		glog.V(1).Infof("Liveness probe unsuccessful: %v", live)
		pb.readinessManager.SetReadiness(containerID, false)
		return live, nil
	}

	// Probe readiness.
	ready, err := pb.probeReadiness(pod, status, container, containerID, createdAt)
	if err == nil && ready == probe.Success {
		glog.V(3).Infof("Readiness probe successful: %v", ready)
		pb.readinessManager.SetReadiness(containerID, true)
		return probe.Success, nil
	}

	glog.V(1).Infof("Readiness probe failed/errored: %v, %v", ready, err)
	pb.readinessManager.SetReadiness(containerID, false)

	ref, ok := pb.refManager.GetRef(containerID)
	if !ok {
		glog.Warningf("No ref for pod '%v' - '%v'", containerID, container.Name)
		return probe.Success, err
	}

	if ready != probe.Success {
		pb.recorder.Eventf(ref, "unhealthy", "Readiness Probe Failed %v - %v", containerID, container.Name)
	}

	return probe.Success, nil
}

// probeLiveness probes the liveness of a container.
// If the initalDelay since container creation on liveness probe has not passed the probe will return probe.Success.
func (pb *prober) probeLiveness(pod *api.Pod, status api.PodStatus, container api.Container, containerID string, createdAt int64) (probe.Result, error) {
	p := container.LivenessProbe
	if p == nil {
		return probe.Success, nil
	}
	if time.Now().Unix()-createdAt < p.InitialDelaySeconds {
		return probe.Success, nil
	}
	return pb.runProbeWithRetries(p, pod, status, container, containerID, maxProbeRetries)
}

// probeReadiness probes the readiness of a container.
// If the initial delay on the readiness probe has not passed the probe will return probe.Failure.
func (pb *prober) probeReadiness(pod *api.Pod, status api.PodStatus, container api.Container, containerID string, createdAt int64) (probe.Result, error) {
	p := container.ReadinessProbe
	if p == nil {
		return probe.Success, nil
	}
	if time.Now().Unix()-createdAt < p.InitialDelaySeconds {
		return probe.Failure, nil
	}
	return pb.runProbeWithRetries(p, pod, status, container, containerID, maxProbeRetries)
}

// runProbeWithRetries tries to probe the container in a finite loop, it returns the last result
// if it never succeeds.
func (pb *prober) runProbeWithRetries(p *api.Probe, pod *api.Pod, status api.PodStatus, container api.Container, containerID string, retries int) (probe.Result, error) {
	var err error
	var result probe.Result
	for i := 0; i < retries; i++ {
		result, err = pb.runProbe(p, pod, status, container, containerID)
		if result == probe.Success {
			return probe.Success, nil
		}
	}
	return result, err
}

func (pb *prober) runProbe(p *api.Probe, pod *api.Pod, status api.PodStatus, container api.Container, containerID string) (probe.Result, error) {
	timeout := time.Duration(p.TimeoutSeconds) * time.Second
	if p.Exec != nil {
		glog.V(4).Infof("Exec-Probe Pod: %v, Container: %v, Command: %v", pod, container, p.Exec.Command)
		return pb.exec.Probe(pb.newExecInContainer(pod, container, containerID, p.Exec.Command))
	}
	if p.HTTPGet != nil {
		port, err := extractPort(p.HTTPGet.Port, container)
		if err != nil {
			return probe.Unknown, err
		}
		host, port, path := extractGetParams(p.HTTPGet, status, port)
		glog.V(4).Infof("HTTP-Probe Host: %v, Port: %v, Path: %v", host, port, path)
		return pb.http.Probe(host, port, path, timeout)
	}
	if p.TCPSocket != nil {
		port, err := extractPort(p.TCPSocket.Port, container)
		if err != nil {
			return probe.Unknown, err
		}
		glog.V(4).Infof("TCP-Probe PodIP: %v, Port: %v, Timeout: %v", status.PodIP, port, timeout)
		return pb.tcp.Probe(status.PodIP, port, timeout)
	}
	glog.Warningf("Failed to find probe builder for container: %v", container)
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

func (p *prober) newExecInContainer(pod *api.Pod, container api.Container, containerID string, cmd []string) exec.Cmd {
	return execInContainer{func() ([]byte, error) {
		return p.Runner.RunInContainer(containerID, cmd)
	}}
}

func (eic execInContainer) CombinedOutput() ([]byte, error) {
	return eic.run()
}

func (eic execInContainer) SetDir(dir string) {
	//unimplemented
}

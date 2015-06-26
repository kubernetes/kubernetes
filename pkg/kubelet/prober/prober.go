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
	"net"
	"net/url"
	"strconv"
	"strings"
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

// Prober helps to check the liveness/readiness of a container.
type prober struct {
	exec   execprobe.ExecProber
	http   httprobe.HTTPProber
	tcp    tcprobe.TCPProber
	runner kubecontainer.ContainerCommandRunner

	readinessManager *kubecontainer.ReadinessManager
	refManager       *kubecontainer.RefManager
	recorder         record.EventRecorder
}

// NewProber creates a Prober, it takes a command runner and
// several container info managers.
func New(
	runner kubecontainer.ContainerCommandRunner,
	readinessManager *kubecontainer.ReadinessManager,
	refManager *kubecontainer.RefManager,
	recorder record.EventRecorder) Prober {

	return &prober{
		exec:   execprobe.New(),
		http:   httprobe.New(),
		tcp:    tcprobe.New(),
		runner: runner,

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
func (pb *prober) Probe(pod *api.Pod, status api.PodStatus, container api.Container, containerID string, createdAt int64) (probe.Result, error) {
	pb.probeReadiness(pod, status, container, containerID, createdAt)
	return pb.probeLiveness(pod, status, container, containerID, createdAt)
}

// probeLiveness probes the liveness of a container.
// If the initalDelay since container creation on liveness probe has not passed the probe will return probe.Success.
func (pb *prober) probeLiveness(pod *api.Pod, status api.PodStatus, container api.Container, containerID string, createdAt int64) (probe.Result, error) {
	var live probe.Result
	var output string
	var err error
	p := container.LivenessProbe
	if p == nil {
		return probe.Success, nil
	}
	if time.Now().Unix()-createdAt < p.InitialDelaySeconds {
		return probe.Success, nil
	} else {
		live, output, err = pb.runProbeWithRetries(p, pod, status, container, containerID, maxProbeRetries)
	}
	ctrName := fmt.Sprintf("%s:%s", kubecontainer.GetPodFullName(pod), container.Name)
	if err != nil || live != probe.Success {
		// Liveness failed in one way or another.
		ref, ok := pb.refManager.GetRef(containerID)
		if !ok {
			glog.Warningf("No ref for pod %q - '%v'", containerID, container.Name)
		}
		if err != nil {
			glog.V(1).Infof("Liveness probe for %q errored: %v", ctrName, err)
			if ok {
				pb.recorder.Eventf(ref, "unhealthy", "Liveness probe errored: %v", err)
			}
			return probe.Unknown, err
		} else { // live != probe.Success
			glog.V(1).Infof("Liveness probe for %q failed (%v): %s", ctrName, live, output)
			if ok {
				pb.recorder.Eventf(ref, "unhealthy", "Liveness probe failed: %s", output)
			}
			return live, nil
		}
	}
	glog.V(3).Infof("Liveness probe for %q succeeded", ctrName)
	return probe.Success, nil
}

// probeReadiness probes and sets the readiness of a container.
// If the initial delay on the readiness probe has not passed, we set readiness to false.
func (pb *prober) probeReadiness(pod *api.Pod, status api.PodStatus, container api.Container, containerID string, createdAt int64) {
	var ready probe.Result
	var output string
	var err error
	p := container.ReadinessProbe
	if p == nil {
		ready = probe.Success
	} else if time.Now().Unix()-createdAt < p.InitialDelaySeconds {
		ready = probe.Failure
	} else {
		ready, output, err = pb.runProbeWithRetries(p, pod, status, container, containerID, maxProbeRetries)
	}
	ctrName := fmt.Sprintf("%s:%s", kubecontainer.GetPodFullName(pod), container.Name)
	if err != nil || ready == probe.Failure {
		// Readiness failed in one way or another.
		pb.readinessManager.SetReadiness(containerID, false)
		ref, ok := pb.refManager.GetRef(containerID)
		if !ok {
			glog.Warningf("No ref for pod '%v' - '%v'", containerID, container.Name)
		}
		if err != nil {
			glog.V(1).Infof("readiness probe for %q errored: %v", ctrName, err)
			if ok {
				pb.recorder.Eventf(ref, "unhealthy", "Readiness probe errored: %v", err)
			}
			return
		} else { // ready != probe.Success
			glog.V(1).Infof("Readiness probe for %q failed (%v): %s", ctrName, ready, output)
			if ok {
				pb.recorder.Eventf(ref, "unhealthy", "Readiness probe failed: %s", output)
			}
			return
		}
	}
	if ready == probe.Success {
		pb.readinessManager.SetReadiness(containerID, true)
	}

	glog.V(3).Infof("Readiness probe for %q succeeded", ctrName)

}

// runProbeWithRetries tries to probe the container in a finite loop, it returns the last result
// if it never succeeds.
func (pb *prober) runProbeWithRetries(p *api.Probe, pod *api.Pod, status api.PodStatus, container api.Container, containerID string, retries int) (probe.Result, string, error) {
	var err error
	var result probe.Result
	var output string
	for i := 0; i < retries; i++ {
		result, output, err = pb.runProbe(p, pod, status, container, containerID)
		if result == probe.Success {
			return probe.Success, output, nil
		}
	}
	return result, output, err
}

func (pb *prober) runProbe(p *api.Probe, pod *api.Pod, status api.PodStatus, container api.Container, containerID string) (probe.Result, string, error) {
	timeout := time.Duration(p.TimeoutSeconds) * time.Second
	if p.Exec != nil {
		glog.V(4).Infof("Exec-Probe Pod: %v, Container: %v, Command: %v", pod, container, p.Exec.Command)
		return pb.exec.Probe(pb.newExecInContainer(pod, container, containerID, p.Exec.Command))
	}
	if p.HTTPGet != nil {
		scheme := strings.ToLower(string(p.HTTPGet.Scheme))
		host := p.HTTPGet.Host
		if host == "" {
			host = status.PodIP
		}
		port, err := extractPort(p.HTTPGet.Port, container)
		if err != nil {
			return probe.Unknown, "", err
		}
		path := p.HTTPGet.Path
		glog.V(4).Infof("HTTP-Probe Host: %v://%v, Port: %v, Path: %v", scheme, host, port, path)
		url := formatURL(scheme, host, port, path)
		return pb.http.Probe(url, timeout)
	}
	if p.TCPSocket != nil {
		port, err := extractPort(p.TCPSocket.Port, container)
		if err != nil {
			return probe.Unknown, "", err
		}
		glog.V(4).Infof("TCP-Probe PodIP: %v, Port: %v, Timeout: %v", status.PodIP, port, timeout)
		return pb.tcp.Probe(status.PodIP, port, timeout)
	}
	glog.Warningf("Failed to find probe builder for container: %v", container)
	return probe.Unknown, "", nil
}

func extractPort(param util.IntOrString, container api.Container) (int, error) {
	port := -1
	var err error
	switch param.Kind {
	case util.IntstrInt:
		port = param.IntVal
	case util.IntstrString:
		if port, err = findPortByName(container, param.StrVal); err != nil {
			// Last ditch effort - maybe it was an int stored as string?
			if port, err = strconv.Atoi(param.StrVal); err != nil {
				return port, err
			}
		}
	default:
		return port, fmt.Errorf("IntOrString had no kind: %+v", param)
	}
	if port > 0 && port < 65536 {
		return port, nil
	}
	return port, fmt.Errorf("invalid port number: %v", port)
}

// findPortByName is a helper function to look up a port in a container by name.
func findPortByName(container api.Container, portName string) (int, error) {
	for _, port := range container.Ports {
		if port.Name == portName {
			return port.HostPort, nil
		}
	}
	return 0, fmt.Errorf("port %s not found", portName)
}

// formatURL formats a URL from args.  For testability.
func formatURL(scheme string, host string, port int, path string) *url.URL {
	return &url.URL{
		Scheme: scheme,
		Host:   net.JoinHostPort(host, strconv.Itoa(port)),
		Path:   path,
	}
}

type execInContainer struct {
	run func() ([]byte, error)
}

func (p *prober) newExecInContainer(pod *api.Pod, container api.Container, containerID string, cmd []string) exec.Cmd {
	return execInContainer{func() ([]byte, error) {
		return p.runner.RunInContainer(containerID, cmd)
	}}
}

func (eic execInContainer) CombinedOutput() ([]byte, error) {
	return eic.run()
}

func (eic execInContainer) SetDir(dir string) {
	//unimplemented
}

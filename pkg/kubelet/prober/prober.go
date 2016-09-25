/*
Copyright 2014 The Kubernetes Authors.

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
	"bytes"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/kubelet/util/ioutils"
	"k8s.io/kubernetes/pkg/probe"
	execprobe "k8s.io/kubernetes/pkg/probe/exec"
	httprobe "k8s.io/kubernetes/pkg/probe/http"
	tcprobe "k8s.io/kubernetes/pkg/probe/tcp"
	"k8s.io/kubernetes/pkg/util/exec"
	"k8s.io/kubernetes/pkg/util/intstr"

	"github.com/golang/glog"
)

const maxProbeRetries = 3

// Prober helps to check the liveness/readiness of a container.
type prober struct {
	exec   execprobe.ExecProber
	http   httprobe.HTTPProber
	tcp    tcprobe.TCPProber
	runner kubecontainer.ContainerCommandRunner

	refManager *kubecontainer.RefManager
	recorder   record.EventRecorder
}

// NewProber creates a Prober, it takes a command runner and
// several container info managers.
func newProber(
	runner kubecontainer.ContainerCommandRunner,
	refManager *kubecontainer.RefManager,
	recorder record.EventRecorder) *prober {

	return &prober{
		exec:       execprobe.New(),
		http:       httprobe.New(),
		tcp:        tcprobe.New(),
		runner:     runner,
		refManager: refManager,
		recorder:   recorder,
	}
}

// probe probes the container.
func (pb *prober) probe(probeType probeType, pod *api.Pod, status api.PodStatus, container api.Container, containerID kubecontainer.ContainerID) (results.Result, error) {
	var probeSpec *api.Probe
	switch probeType {
	case readiness:
		probeSpec = container.ReadinessProbe
	case liveness:
		probeSpec = container.LivenessProbe
	default:
		return results.Failure, fmt.Errorf("Unknown probe type: %q", probeType)
	}

	ctrName := fmt.Sprintf("%s:%s", format.Pod(pod), container.Name)
	if probeSpec == nil {
		glog.Warningf("%s probe for %s is nil", probeType, ctrName)
		return results.Success, nil
	}

	result, output, err := pb.runProbeWithRetries(probeSpec, pod, status, container, containerID, maxProbeRetries)
	if err != nil || result != probe.Success {
		// Probe failed in one way or another.
		ref, hasRef := pb.refManager.GetRef(containerID)
		if !hasRef {
			glog.Warningf("No ref for container %q (%s)", containerID.String(), ctrName)
		}
		if err != nil {
			glog.V(1).Infof("%s probe for %q errored: %v", probeType, ctrName, err)
			if hasRef {
				pb.recorder.Eventf(ref, api.EventTypeWarning, events.ContainerUnhealthy, "%s probe errored: %v", probeType, err)
			}
		} else { // result != probe.Success
			glog.V(1).Infof("%s probe for %q failed (%v): %s", probeType, ctrName, result, output)
			if hasRef {
				pb.recorder.Eventf(ref, api.EventTypeWarning, events.ContainerUnhealthy, "%s probe failed: %s", probeType, output)
			}
		}
		return results.Failure, err
	}
	glog.V(3).Infof("%s probe for %q succeeded", probeType, ctrName)
	return results.Success, nil
}

// runProbeWithRetries tries to probe the container in a finite loop, it returns the last result
// if it never succeeds.
func (pb *prober) runProbeWithRetries(p *api.Probe, pod *api.Pod, status api.PodStatus, container api.Container, containerID kubecontainer.ContainerID, retries int) (probe.Result, string, error) {
	var err error
	var result probe.Result
	var output string
	for i := 0; i < retries; i++ {
		result, output, err = pb.runProbe(p, pod, status, container, containerID)
		if err == nil {
			return result, output, nil
		}
	}
	return result, output, err
}

// buildHeaderMap takes a list of HTTPHeader <name, value> string
// pairs and returns a populated string->[]string http.Header map.
func buildHeader(headerList []api.HTTPHeader) http.Header {
	headers := make(http.Header)
	for _, header := range headerList {
		headers[header.Name] = append(headers[header.Name], header.Value)
	}
	return headers
}

func (pb *prober) runProbe(p *api.Probe, pod *api.Pod, status api.PodStatus, container api.Container, containerID kubecontainer.ContainerID) (probe.Result, string, error) {
	timeout := time.Duration(p.TimeoutSeconds) * time.Second
	if p.Exec != nil {
		glog.V(4).Infof("Exec-Probe Pod: %v, Container: %v, Command: %v", pod, container, p.Exec.Command)
		return pb.exec.Probe(pb.newExecInContainer(container, containerID, p.Exec.Command))
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
		headers := buildHeader(p.HTTPGet.HTTPHeaders)
		glog.V(4).Infof("HTTP-Probe Headers: %v", headers)
		return pb.http.Probe(url, headers, timeout)
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
	return probe.Unknown, "", fmt.Errorf("Missing probe handler for %s:%s", format.Pod(pod), container.Name)
}

func extractPort(param intstr.IntOrString, container api.Container) (int, error) {
	port := -1
	var err error
	switch param.Type {
	case intstr.Int:
		port = param.IntValue()
	case intstr.String:
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
			return int(port.ContainerPort), nil
		}
	}
	return 0, fmt.Errorf("port %s not found", portName)
}

// formatURL formats a URL from args.  For testability.
func formatURL(scheme string, host string, port int, path string) *url.URL {
	u, err := url.Parse(path)
	// Something is busted with the path, but it's too late to reject it. Pass it along as is.
	if err != nil {
		u = &url.URL{
			Path: path,
		}
	}
	u.Scheme = scheme
	u.Host = net.JoinHostPort(host, strconv.Itoa(port))
	return u
}

type execInContainer struct {
	// run executes a command in a container. Combined stdout and stderr output is always returned. An
	// error is returned if one occurred.
	run func() ([]byte, error)
}

func (p *prober) newExecInContainer(container api.Container, containerID kubecontainer.ContainerID, cmd []string) exec.Cmd {
	return execInContainer{func() ([]byte, error) {
		var buffer bytes.Buffer
		output := ioutils.WriteCloserWrapper(&buffer)
		err := p.runner.ExecInContainer(containerID, cmd, nil, output, output, false, nil)
		// Even if err is non-nil, there still may be output (e.g. the exec wrote to stdout or stderr but
		// the command returned a nonzero exit code). Therefore, always return the output along with the
		// error.
		return buffer.Bytes(), err
	}}
}

func (eic execInContainer) CombinedOutput() ([]byte, error) {
	return eic.run()
}

func (eic execInContainer) Output() ([]byte, error) {
	return nil, fmt.Errorf("unimplemented")
}

func (eic execInContainer) SetDir(dir string) {
	//unimplemented
}

func (eic execInContainer) SetStdin(in io.Reader) {
	//unimplemented
}

func (eic execInContainer) SetStdout(out io.Writer) {
	//unimplemented
}

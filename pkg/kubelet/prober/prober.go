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
	"context"
	"fmt"
	"io"
	"net/http"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/probe"
	execprobe "k8s.io/kubernetes/pkg/probe/exec"
	grpcprobe "k8s.io/kubernetes/pkg/probe/grpc"
	httpprobe "k8s.io/kubernetes/pkg/probe/http"
	tcpprobe "k8s.io/kubernetes/pkg/probe/tcp"
	"k8s.io/utils/exec"

	"k8s.io/klog/v2"
)

const maxProbeRetries = 3

// Prober helps to check the liveness/readiness/startup of a container.
type prober struct {
	exec   execprobe.Prober
	http   httpprobe.Prober
	tcp    tcpprobe.Prober
	grpc   grpcprobe.Prober
	runner kubecontainer.CommandRunner

	recorder record.EventRecorderLogger
}

// NewProber creates a Prober, it takes a command runner and
// several container info managers.
func newProber(
	runner kubecontainer.CommandRunner,
	recorder record.EventRecorderLogger) *prober {

	const followNonLocalRedirects = false
	return &prober{
		exec:     execprobe.New(),
		http:     httpprobe.New(followNonLocalRedirects),
		tcp:      tcpprobe.New(),
		grpc:     grpcprobe.New(),
		runner:   runner,
		recorder: recorder,
	}
}

// recordContainerEvent should be used by the prober for all container related events.
func (pb *prober) recordContainerEvent(ctx context.Context, pod *v1.Pod, container *v1.Container, eventType, reason, message string, args ...interface{}) {
	logger := klog.FromContext(ctx)
	ref, err := kubecontainer.GenerateContainerRef(pod, container)
	if err != nil {
		logger.Error(err, "Can't make a ref to pod and container", "pod", klog.KObj(pod), "containerName", container.Name)
		return
	}
	pb.recorder.WithLogger(logger).Eventf(ref, eventType, reason, message, args...)
}

// probe probes the container.
func (pb *prober) probe(ctx context.Context, action ProberAction, probeType probeType, pod *v1.Pod, status v1.PodStatus, container v1.Container, containerID kubecontainer.ContainerID) (results.Result, error) {
	var probeSpec *v1.Probe
	switch probeType {
	case readiness:
		probeSpec = container.ReadinessProbe
	case liveness:
		probeSpec = container.LivenessProbe
	case startup:
		probeSpec = container.StartupProbe
	default:
		return results.Failure, fmt.Errorf("unknown probe type: %q", probeType)
	}

	logger := klog.FromContext(ctx)
	if probeSpec == nil {
		logger.Info("Probe is nil", "probeType", probeType, "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", container.Name)
		return results.Success, nil
	}

	result, output, err := pb.runProbeWithRetries(ctx, action, status, containerID, maxProbeRetries)

	if err != nil {
		// Handle probe error
		logger.V(1).Info("Probe errored", "probeType", probeType, "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", container.Name, "probeResult", result, "err", err)
		pb.recordContainerEvent(ctx, pod, &container, v1.EventTypeWarning, events.ContainerUnhealthy, "%s probe errored and resulted in %s state: %s", probeType, result, err)
		return results.Failure, err
	}

	switch result {
	case probe.Success:
		logger.V(3).Info("Probe succeeded", "probeType", probeType, "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", container.Name)
		return results.Success, nil

	case probe.Warning:
		pb.recordContainerEvent(ctx, pod, &container, v1.EventTypeWarning, events.ContainerProbeWarning, "%s probe warning: %s", probeType, output)
		logger.V(3).Info("Probe succeeded with a warning", "probeType", probeType, "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", container.Name, "output", output)
		return results.Success, nil

	case probe.Failure:
		logger.V(1).Info("Probe failed", "probeType", probeType, "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", container.Name, "probeResult", result, "output", output)
		pb.recordContainerEvent(ctx, pod, &container, v1.EventTypeWarning, events.ContainerUnhealthy, "%s probe failed: %s", probeType, output)
		return results.Failure, nil

	case probe.Unknown:
		logger.V(1).Info("Probe unknown without error", "probeType", probeType, "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", container.Name, "probeResult", result)
		return results.Failure, nil

	default:
		logger.V(1).Info("Unsupported probe result", "probeType", probeType, "pod", klog.KObj(pod), "podUID", pod.UID, "containerName", container.Name, "probeResult", result)
		return results.Failure, nil
	}
}

// ProberAction encapsulates the logic of a probe operation.
type ProberAction interface {
	Run(ctx context.Context, status v1.PodStatus, containerID kubecontainer.ContainerID) (probe.Result, string, error)
}

func (pb *prober) newProberAction(probeType probeType, p *v1.Probe, pod *v1.Pod, container v1.Container) ProberAction {
	if p == nil {
		return nil
	}
	switch {
	case p.Exec != nil:
		return &execProberAction{pb: pb, pod: pod, container: container, p: p}
	case p.HTTPGet != nil:
		return &httpProberAction{pb: pb, probeType: probeType, pod: pod, container: container, p: p}
	case p.TCPSocket != nil:
		return &tcpProberAction{pb: pb, container: container, p: p}
	case p.GRPC != nil:
		return &grpcProberAction{pb: pb, p: p}
	default:
		return &unknownProberAction{pod: pod, container: container}
	}
}

type execProberAction struct {
	pb        *prober
	pod       *v1.Pod
	container v1.Container
	p         *v1.Probe
}

func (a *execProberAction) Run(ctx context.Context, _ v1.PodStatus, containerID kubecontainer.ContainerID) (probe.Result, string, error) {
	logger := klog.FromContext(ctx)
	timeout := time.Duration(a.p.TimeoutSeconds) * time.Second
	logger.V(4).Info("Exec-Probe runProbe", "pod", klog.KObj(a.pod), "containerName", a.container.Name, "execCommand", a.p.Exec.Command)
	command := kubecontainer.ExpandContainerCommandOnlyStatic(a.p.Exec.Command, a.container.Env)
	return a.pb.exec.Probe(a.pb.newExecInContainer(ctx, a.pod, a.container, containerID, command, timeout))
}

type httpProberAction struct {
	pb        *prober
	probeType probeType
	pod       *v1.Pod
	container v1.Container
	p         *v1.Probe

	req *http.Request
}

func (a *httpProberAction) Run(ctx context.Context, status v1.PodStatus, _ kubecontainer.ContainerID) (probe.Result, string, error) {
	logger := klog.FromContext(ctx)
	timeout := time.Duration(a.p.TimeoutSeconds) * time.Second

	req := a.req
	hostFromProbe := a.p.HTTPGet.Host
	if req == nil || (hostFromProbe == "" && req.URL.Hostname() != status.PodIP) {
		var err error
		req, err = httpprobe.NewRequestForHTTPGetAction(a.p.HTTPGet, &a.container, status.PodIP, "probe")
		if err != nil {
			logger.V(4).Info("HTTP-Probe failed to create request", "error", err)
			return probe.Unknown, "", err
		}
		if status.PodIP != "" {
			a.req = req
		}
	}

	if loggerV4 := logger.V(4); loggerV4.Enabled() {
		port := req.URL.Port()
		host := req.URL.Hostname()
		path := req.URL.Path
		scheme := req.URL.Scheme
		headers := a.p.HTTPGet.HTTPHeaders
		loggerV4.Info("HTTP-Probe", "scheme", scheme, "host", host, "port", port, "path", path, "timeout", timeout, "headers", headers, "probeType", a.probeType)
	}

	reqWithCtx := req.WithContext(ctx)
	return a.pb.http.Probe(reqWithCtx, timeout)
}

type tcpProberAction struct {
	pb        *prober
	container v1.Container
	p         *v1.Probe
}

func (a *tcpProberAction) Run(ctx context.Context, status v1.PodStatus, _ kubecontainer.ContainerID) (probe.Result, string, error) {
	logger := klog.FromContext(ctx)
	timeout := time.Duration(a.p.TimeoutSeconds) * time.Second
	port, err := probe.ResolveContainerPort(a.p.TCPSocket.Port, &a.container)
	if err != nil {
		logger.V(4).Info("TCP-Probe failed to resolve port", "error", err)
		return probe.Unknown, "", err
	}
	host := a.p.TCPSocket.Host
	if host == "" {
		host = status.PodIP
	}
	logger.V(4).Info("TCP-Probe", "host", host, "port", port, "timeout", timeout)
	return a.pb.tcp.Probe(host, port, timeout)
}

type grpcProberAction struct {
	pb *prober
	p  *v1.Probe
}

func (a *grpcProberAction) Run(ctx context.Context, status v1.PodStatus, _ kubecontainer.ContainerID) (probe.Result, string, error) {
	logger := klog.FromContext(ctx)
	timeout := time.Duration(a.p.TimeoutSeconds) * time.Second
	host := status.PodIP
	service := ""
	if a.p.GRPC.Service != nil {
		service = *a.p.GRPC.Service
	}
	logger.V(4).Info("GRPC-Probe", "host", host, "service", service, "port", a.p.GRPC.Port, "timeout", timeout)
	return a.pb.grpc.Probe(host, service, int(a.p.GRPC.Port), timeout)
}

type unknownProberAction struct {
	pod       *v1.Pod
	container v1.Container
}

func (a *unknownProberAction) Run(ctx context.Context, _ v1.PodStatus, _ kubecontainer.ContainerID) (probe.Result, string, error) {
	logger := klog.FromContext(ctx)
	logger.V(4).Info("Failed to find probe builder for container", "containerName", a.container.Name)
	return probe.Unknown, "", fmt.Errorf("missing probe handler for %s:%s", format.Pod(a.pod), a.container.Name)
}

// runProbeWithRetries tries to probe the container in a finite loop, it returns the last result
// if it never succeeds.
func (pb *prober) runProbeWithRetries(ctx context.Context, action ProberAction, status v1.PodStatus, containerID kubecontainer.ContainerID, retries int) (probe.Result, string, error) {
	var err error
	var result probe.Result
	var output string
	for range retries {
		result, output, err = action.Run(ctx, status, containerID)
		if err == nil {
			return result, output, nil
		}
	}
	return result, output, err
}

type execInContainer struct {
	// run executes a command in a container. Combined stdout and stderr output is always returned. An
	// error is returned if one occurred.
	run       func() ([]byte, error)
	writer    io.Writer
	pod       *v1.Pod
	container v1.Container
}

func (pb *prober) newExecInContainer(ctx context.Context, pod *v1.Pod, container v1.Container, containerID kubecontainer.ContainerID, cmd []string, timeout time.Duration) exec.Cmd {
	return &execInContainer{
		run:       func() ([]byte, error) { return pb.runner.RunInContainer(ctx, containerID, cmd, timeout) },
		pod:       pod,
		container: container,
	}
}

func (eic *execInContainer) Run() error {
	return nil
}

func (eic *execInContainer) CombinedOutput() ([]byte, error) {
	return eic.run()
}

func (eic *execInContainer) Output() ([]byte, error) {
	return nil, fmt.Errorf("unimplemented")
}

func (eic *execInContainer) SetDir(dir string) {
	// unimplemented
}

func (eic *execInContainer) SetStdin(in io.Reader) {
	// unimplemented
}

func (eic *execInContainer) SetStdout(out io.Writer) {
	eic.writer = out
}

func (eic *execInContainer) SetStderr(out io.Writer) {
	eic.writer = out
}

func (eic *execInContainer) SetEnv(env []string) {
	// unimplemented
}

func (eic *execInContainer) Stop() {
	// unimplemented
}

func (eic *execInContainer) Start() error {
	data, err := eic.run()
	if eic.writer != nil {
		// only record the write error, do not cover the command run error
		if p, err := eic.writer.Write(data); err != nil {
			// Use klog.TODO() because we currently do not have a proper context/logger to pass in.
			// Replace this with an appropriate context/logger when refactoring this function to accept a context parameter.
			klog.TODO().Error(err, "Unable to write all bytes from execInContainer", "expectedBytes", len(data), "actualBytes", p, "pod", klog.KObj(eic.pod), "containerName", eic.container.Name)
		}
	}
	return err
}

func (eic *execInContainer) Wait() error {
	return nil
}

func (eic *execInContainer) StdoutPipe() (io.ReadCloser, error) {
	return nil, fmt.Errorf("unimplemented")
}

func (eic *execInContainer) StderrPipe() (io.ReadCloser, error) {
	return nil, fmt.Errorf("unimplemented")
}

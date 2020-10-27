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

package lifecycle

import (
	"fmt"
	"net"
	"net/http"
	"strconv"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/security/apparmor"
	utilio "k8s.io/utils/io"
)

const (
	maxRespBodyLength = 10 * 1 << 10 // 10KB
)

type handlerRunner struct {
	httpGetter       kubetypes.HTTPGetter
	commandRunner    kubecontainer.CommandRunner
	containerManager podStatusProvider
}

type podStatusProvider interface {
	GetPodStatus(uid types.UID, name, namespace string) (*kubecontainer.PodStatus, error)
}

// NewHandlerRunner returns a configured lifecycle handler for a container.
func NewHandlerRunner(httpGetter kubetypes.HTTPGetter, commandRunner kubecontainer.CommandRunner, containerManager podStatusProvider) kubecontainer.HandlerRunner {
	return &handlerRunner{
		httpGetter:       httpGetter,
		commandRunner:    commandRunner,
		containerManager: containerManager,
	}
}

func (hr *handlerRunner) Run(containerID kubecontainer.ContainerID, pod *v1.Pod, container *v1.Container, handler *v1.Handler) (string, error) {
	switch {
	case handler.Exec != nil:
		var msg string
		// TODO(tallclair): Pass a proper timeout value.
		output, err := hr.commandRunner.RunInContainer(containerID, handler.Exec.Command, 0)
		if err != nil {
			msg = fmt.Sprintf("Exec lifecycle hook (%v) for Container %q in Pod %q failed - error: %v, message: %q", handler.Exec.Command, container.Name, format.Pod(pod), err, string(output))
			klog.V(1).Infof(msg)
		}
		return msg, err
	case handler.HTTPGet != nil:
		msg, err := hr.runHTTPHandler(pod, container, handler)
		if err != nil {
			msg = fmt.Sprintf("Http lifecycle hook (%s) for Container %q in Pod %q failed - error: %v, message: %q", handler.HTTPGet.Path, container.Name, format.Pod(pod), err, msg)
			klog.V(1).Infof(msg)
		}
		return msg, err
	default:
		err := fmt.Errorf("invalid handler: %v", handler)
		msg := fmt.Sprintf("Cannot run handler: %v", err)
		klog.Errorf(msg)
		return msg, err
	}
}

// resolvePort attempts to turn an IntOrString port reference into a concrete port number.
// If portReference has an int value, it is treated as a literal, and simply returns that value.
// If portReference is a string, an attempt is first made to parse it as an integer.  If that fails,
// an attempt is made to find a port with the same name in the container spec.
// If a port with the same name is found, it's ContainerPort value is returned.  If no matching
// port is found, an error is returned.
func resolvePort(portReference intstr.IntOrString, container *v1.Container) (int, error) {
	if portReference.Type == intstr.Int {
		return portReference.IntValue(), nil
	}
	portName := portReference.StrVal
	port, err := strconv.Atoi(portName)
	if err == nil {
		return port, nil
	}
	for _, portSpec := range container.Ports {
		if portSpec.Name == portName {
			return int(portSpec.ContainerPort), nil
		}
	}
	return -1, fmt.Errorf("couldn't find port: %v in %v", portReference, container)
}

func (hr *handlerRunner) runHTTPHandler(pod *v1.Pod, container *v1.Container, handler *v1.Handler) (string, error) {
	host := handler.HTTPGet.Host
	if len(host) == 0 {
		status, err := hr.containerManager.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
		if err != nil {
			klog.Errorf("Unable to get pod info, event handlers may be invalid.")
			return "", err
		}
		if len(status.IPs) == 0 {
			return "", fmt.Errorf("failed to find networking container: %v", status)
		}
		host = status.IPs[0]
	}
	var port int
	if handler.HTTPGet.Port.Type == intstr.String && len(handler.HTTPGet.Port.StrVal) == 0 {
		port = 80
	} else {
		var err error
		port, err = resolvePort(handler.HTTPGet.Port, container)
		if err != nil {
			return "", err
		}
	}
	url := fmt.Sprintf("http://%s/%s", net.JoinHostPort(host, strconv.Itoa(port)), handler.HTTPGet.Path)
	resp, err := hr.httpGetter.Get(url)
	return getHTTPRespBody(resp), err
}

func getHTTPRespBody(resp *http.Response) string {
	if resp == nil {
		return ""
	}
	defer resp.Body.Close()
	bytes, err := utilio.ReadAtMost(resp.Body, maxRespBodyLength)
	if err == nil || err == utilio.ErrLimitReached {
		return string(bytes)
	}
	return ""
}

// NewAppArmorAdmitHandler returns a PodAdmitHandler which is used to evaluate
// if a pod can be admitted from the perspective of AppArmor.
func NewAppArmorAdmitHandler(validator apparmor.Validator) PodAdmitHandler {
	return &appArmorAdmitHandler{
		Validator: validator,
	}
}

type appArmorAdmitHandler struct {
	apparmor.Validator
}

func (a *appArmorAdmitHandler) Admit(attrs *PodAdmitAttributes) PodAdmitResult {
	// If the pod is already running or terminated, no need to recheck AppArmor.
	if attrs.Pod.Status.Phase != v1.PodPending {
		return PodAdmitResult{Admit: true}
	}

	err := a.Validate(attrs.Pod)
	if err == nil {
		return PodAdmitResult{Admit: true}
	}
	return PodAdmitResult{
		Admit:   false,
		Reason:  "AppArmor",
		Message: fmt.Sprintf("Cannot enforce AppArmor: %v", err),
	}
}

// NewNoNewPrivsAdmitHandler returns a PodAdmitHandler which is used to evaluate
// if a pod can be admitted from the perspective of NoNewPrivs.
func NewNoNewPrivsAdmitHandler(runtime kubecontainer.Runtime) PodAdmitHandler {
	return &noNewPrivsAdmitHandler{
		Runtime: runtime,
	}
}

type noNewPrivsAdmitHandler struct {
	kubecontainer.Runtime
}

func (a *noNewPrivsAdmitHandler) Admit(attrs *PodAdmitAttributes) PodAdmitResult {
	// If the pod is already running or terminated, no need to recheck NoNewPrivs.
	if attrs.Pod.Status.Phase != v1.PodPending {
		return PodAdmitResult{Admit: true}
	}

	// If the containers in a pod do not require no-new-privs, admit it.
	if !noNewPrivsRequired(attrs.Pod) {
		return PodAdmitResult{Admit: true}
	}

	// Always admit runtimes except docker.
	if a.Runtime.Type() != kubetypes.DockerContainerRuntime {
		return PodAdmitResult{Admit: true}
	}

	// Make sure docker api version is valid.
	rversion, err := a.Runtime.APIVersion()
	if err != nil {
		return PodAdmitResult{
			Admit:   false,
			Reason:  "NoNewPrivs",
			Message: fmt.Sprintf("Cannot enforce NoNewPrivs: %v", err),
		}
	}
	v, err := rversion.Compare("1.23.0")
	if err != nil {
		return PodAdmitResult{
			Admit:   false,
			Reason:  "NoNewPrivs",
			Message: fmt.Sprintf("Cannot enforce NoNewPrivs: %v", err),
		}
	}
	// If the version is less than 1.23 it will return -1 above.
	if v == -1 {
		return PodAdmitResult{
			Admit:   false,
			Reason:  "NoNewPrivs",
			Message: fmt.Sprintf("Cannot enforce NoNewPrivs: docker runtime API version %q must be greater than or equal to 1.23", rversion.String()),
		}
	}

	return PodAdmitResult{Admit: true}
}

func noNewPrivsRequired(pod *v1.Pod) bool {
	// Iterate over pod containers and check if we added no-new-privs.
	for _, c := range pod.Spec.Containers {
		if c.SecurityContext != nil && c.SecurityContext.AllowPrivilegeEscalation != nil && !*c.SecurityContext.AllowPrivilegeEscalation {
			return true
		}
	}
	return false
}

// NewProcMountAdmitHandler returns a PodAdmitHandler which is used to evaluate
// if a pod can be admitted from the perspective of ProcMount.
func NewProcMountAdmitHandler(runtime kubecontainer.Runtime) PodAdmitHandler {
	return &procMountAdmitHandler{
		Runtime: runtime,
	}
}

type procMountAdmitHandler struct {
	kubecontainer.Runtime
}

func (a *procMountAdmitHandler) Admit(attrs *PodAdmitAttributes) PodAdmitResult {
	// If the pod is already running or terminated, no need to recheck NoNewPrivs.
	if attrs.Pod.Status.Phase != v1.PodPending {
		return PodAdmitResult{Admit: true}
	}

	// If the containers in a pod only need the default ProcMountType, admit it.
	if procMountIsDefault(attrs.Pod) {
		return PodAdmitResult{Admit: true}
	}

	// Always admit runtimes except docker.
	if a.Runtime.Type() != kubetypes.DockerContainerRuntime {
		return PodAdmitResult{Admit: true}
	}

	// Make sure docker api version is valid.
	// Merged in https://github.com/moby/moby/pull/36644
	rversion, err := a.Runtime.APIVersion()
	if err != nil {
		return PodAdmitResult{
			Admit:   false,
			Reason:  "ProcMount",
			Message: fmt.Sprintf("Cannot enforce ProcMount: %v", err),
		}
	}
	v, err := rversion.Compare("1.38.0")
	if err != nil {
		return PodAdmitResult{
			Admit:   false,
			Reason:  "ProcMount",
			Message: fmt.Sprintf("Cannot enforce ProcMount: %v", err),
		}
	}
	// If the version is less than 1.38 it will return -1 above.
	if v == -1 {
		return PodAdmitResult{
			Admit:   false,
			Reason:  "ProcMount",
			Message: fmt.Sprintf("Cannot enforce ProcMount: docker runtime API version %q must be greater than or equal to 1.38", rversion.String()),
		}
	}

	return PodAdmitResult{Admit: true}
}

func procMountIsDefault(pod *v1.Pod) bool {
	// Iterate over pod containers and check if we are using the DefaultProcMountType
	// for all containers.
	for _, c := range pod.Spec.Containers {
		if c.SecurityContext != nil {
			if c.SecurityContext.ProcMount != nil && *c.SecurityContext.ProcMount != v1.DefaultProcMount {
				return false
			}
		}
	}

	return true
}

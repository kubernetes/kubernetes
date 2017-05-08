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
	"io/ioutil"
	"net"
	"net/http"
	"strconv"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/security/apparmor"
)

type HandlerRunner struct {
	httpGetter       kubetypes.HttpGetter
	commandRunner    kubecontainer.ContainerCommandRunner
	containerManager podStatusProvider
}

type podStatusProvider interface {
	GetPodStatus(uid types.UID, name, namespace string) (*kubecontainer.PodStatus, error)
}

func NewHandlerRunner(httpGetter kubetypes.HttpGetter, commandRunner kubecontainer.ContainerCommandRunner, containerManager podStatusProvider) kubecontainer.HandlerRunner {
	return &HandlerRunner{
		httpGetter:       httpGetter,
		commandRunner:    commandRunner,
		containerManager: containerManager,
	}
}

func (hr *HandlerRunner) RunPostStart(containerID kubecontainer.ContainerID, pod *v1.Pod, container *v1.Container, handler *v1.Handler) (string, error) {
	switch {
	case handler.Exec != nil:
		return hr.runExec(containerID, pod, container, handler.Exec.Command)
	case handler.HTTPGet != nil:
		return hr.runHTTP(containerID, pod, container, handler.HTTPGet)
	default:
		err := fmt.Errorf("Invalid handler: %v", handler)
		msg := fmt.Sprintf("Cannot run handler: %v", err)
		glog.Errorf(msg)
		return msg, err
	}
}

func (hr *HandlerRunner) RunPreStop(containerID kubecontainer.ContainerID, pod *v1.Pod, container *v1.Container, handler *v1.PreStopHandler) (string, error) {
	switch {
	case handler.Exec != nil:
		return hr.runExec(containerID, pod, container, handler.Exec.Command)
	case handler.HTTPGet != nil:
		return hr.runHTTP(containerID, pod, container, &handler.HTTPGet.HTTPGetAction)
	default:
		err := fmt.Errorf("Invalid handler: %v", handler)
		msg := fmt.Sprintf("Cannot run handler: %v", err)
		glog.Errorf(msg)
		return msg, err
	}
}

func (hr *HandlerRunner) runExec(containerID kubecontainer.ContainerID, pod *v1.Pod, container *v1.Container, command []string) (string, error) {
	// TODO(timstclair): Pass a proper timeout value.
	output, err := hr.commandRunner.RunInContainer(containerID, command, 0)
	if err != nil {
		glog.V(1).Infof("Exec lifecycle hook (%v) for Container %q in Pod %q failed - error: %v, message: %q", command, container.Name, format.Pod(pod), err, string(output))
	}
	return string(output), err
}

func (hr *HandlerRunner) runHTTP(containerID kubecontainer.ContainerID, pod *v1.Pod, container *v1.Container, action *v1.HTTPGetAction) (string, error) {
	msg, err := hr.runHTTPHandler(pod, container, action)
	if err != nil {
		glog.V(1).Infof("Http lifecycle hook (%s) for Container %q in Pod %q failed - error: %v, message: %q", action.Path, container.Name, format.Pod(pod), err, msg)
	}
	return msg, err
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

func (hr *HandlerRunner) runHTTPHandler(pod *v1.Pod, container *v1.Container, action *v1.HTTPGetAction) (string, error) {
	host := action.Host
	if len(host) == 0 {
		status, err := hr.containerManager.GetPodStatus(pod.UID, pod.Name, pod.Namespace)
		if err != nil {
			glog.Errorf("Unable to get pod info, event handlers may be invalid.")
			return "", err
		}
		if status.IP == "" {
			return "", fmt.Errorf("failed to find networking container: %v", status)
		}
		host = status.IP
	}
	var port int
	if action.Port.Type == intstr.String && len(action.Port.StrVal) == 0 {
		port = 80
	} else {
		var err error
		port, err = resolvePort(action.Port, container)
		if err != nil {
			return "", err
		}
	}
	url := fmt.Sprintf("http://%s/%s", net.JoinHostPort(host, strconv.Itoa(port)), action.Path)
	resp, err := hr.httpGetter.Get(url)
	return getHttpRespBody(resp), err
}

func getHttpRespBody(resp *http.Response) string {
	if resp == nil {
		return ""
	}
	defer resp.Body.Close()
	if bytes, err := ioutil.ReadAll(resp.Body); err == nil {
		return string(bytes)
	}
	return ""
}

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

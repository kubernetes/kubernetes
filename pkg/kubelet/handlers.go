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
	"net"
	"strconv"
	"net/http"
	"io"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type execActionHandler struct {
	kubelet *Kubelet
}

func (e *execActionHandler) Run(podFullName, uuid string, container *api.Container, handler *api.Handler) error {
	_, err := e.kubelet.RunInContainer(podFullName, uuid, container.Name, handler.Exec.Command)
	return err
}

type httpActionHandler struct {
	kubelet *Kubelet
	client  httpGetInterface
}

// ResolvePort attempts to turn a IntOrString port reference into a concrete port number.
// If portReference has an int value, it is treated as a literal, and simply returns that value.
// If portReference is a string, an attempt is first made to parse it as an integer.  If that fails,
// an attempt is made to find a port with the same name in the container spec.
// If a port with the same name is found, it's ContainerPort value is returned.  If no matching
// port is found, an error is returned.
func ResolvePort(portReference util.IntOrString, container *api.Container) (int, error) {
	if portReference.Kind == util.IntstrInt {
		return portReference.IntVal, nil
	} else {
		portName := portReference.StrVal
		port, err := strconv.Atoi(portName)
		if err == nil {
			return port, nil
		}
		for _, portSpec := range container.Ports {
			if portSpec.Name == portName {
				return portSpec.ContainerPort, nil
			}
		}

	}
	return -1, fmt.Errorf("couldn't find port: %v in %v", portReference, container)
}

func (h *httpActionHandler) Run(podFullName, uuid string, container *api.Container, handler *api.Handler) error {
	host := handler.HTTPGet.Host
	if len(host) == 0 {
		var info api.PodInfo
		info, err := h.kubelet.GetPodInfo(podFullName, uuid)
		if err != nil {
			glog.Errorf("unable to get pod info, event handlers may be invalid.")
			return err
		}
		netInfo, found := info[networkContainerName]
		if found && netInfo.NetworkSettings != nil {
			host = netInfo.NetworkSettings.IPAddress
		} else {
			return fmt.Errorf("failed to find networking container: %v", info)
		}
	}
	var port int
	if handler.HTTPGet.Port.Kind == util.IntstrString && len(handler.HTTPGet.Port.StrVal) == 0 {
		port = 80
	} else {
		var err error
		port, err = ResolvePort(handler.HTTPGet.Port, container)
		if err != nil {
			return err
		}
	}
	url := fmt.Sprintf("http://%s/%s", net.JoinHostPort(host, strconv.Itoa(port)), handler.HTTPGet.Path)
	_, err := h.client.Get(url)
	return err
}

// FlushWriter provides wrapper for responseWriter with HTTP streaming capabilities
type FlushWriter struct {
	flusher http.Flusher
	writer io.Writer
}

// Write is a FlushWriter implementation of the io.Writer that sends any buffered data to the client.
func (fw *FlushWriter) Write(p []byte) (n int, err error) {
	n, err = fw.writer.Write(p)
	if err != nil {
		return
	}
	if fw.flusher != nil {
		fw.flusher.Flush()
	}
	return
}

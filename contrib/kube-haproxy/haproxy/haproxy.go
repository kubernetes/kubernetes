/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package haproxy

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/exec"

	"github.com/golang/glog"
)

type HAProxyManager struct {
	Exec           exec.Interface
	ConfigFile     string
	HTTPPort       int
	simpleServices []serviceRecord
	routedServices map[string]*serviceRecord
}

type serviceRecord struct {
	namespace   string
	serviceName string
	lastVersion string
	service     *api.Service
	endpoints   *api.Endpoints
}

func findPortInSubset(subset *api.EndpointSubset, port *api.ServicePort) *api.EndpointPort {
	for ix := range subset.Ports {
		subsetPort := &subset.Ports[ix]
		switch port.TargetPort.Kind {
		case util.IntstrInt:
			if subsetPort.Port == port.TargetPort.IntVal {
				return subsetPort
			}
		case util.IntstrString:
			if subsetPort.Name == port.TargetPort.StrVal {
				return subsetPort
			}
		}
	}
	return nil
}

func writeSimpleService(service *api.Service, servicePort *api.ServicePort, endpoints *api.Endpoints, mode string, writer io.Writer) {
	/*
		Example:
			listen nginx :80
			    mode tcp
				balance leastconn
				server node-1 10.176.0.5:80 check
				server node-2 10.176.1.4:80 check
	*/
	fmt.Fprintf(writer, "listen %s :%d\n", service.Name, servicePort.Port)
	fmt.Fprintf(writer, "    mode tcp\n")
	fmt.Fprintf(writer, "    balance leastconn\n")
	for _, subset := range endpoints.Subsets {
		port := findPortInSubset(&subset, servicePort)
		if port != nil {
			for _, address := range subset.Addresses {
				fmt.Fprintf(writer, "    server %s %s:%d check\n", address.TargetRef.Name, address.IP, port.Port)
			}
		}
	}
	fmt.Fprintf(writer, "\n")
}

func (h *HAProxyManager) writeHTTPFrontend(writer io.Writer) {
	/*
		Example:
			frontend www
				bind :8080
				acl backend1 /some/path path_beg
				acl backend2 /other/path path_beg
	*/
	fmt.Fprintf(writer, "frontend www\n")
	fmt.Fprintf(writer, "    bind :%d\n", h.HTTPPort)
	for path, record := range h.routedServices {
		fmt.Fprintf(writer, "    acl %s path_beg %s\n", record.serviceName, path)
	}
	for _, record := range h.routedServices {
		fmt.Fprintf(writer, "    use_backend %s if %s\n", record.serviceName, record.serviceName)
	}
	fmt.Fprintf(writer, "\n")
}

func (h *HAProxyManager) writeHTTPBackend(path string, record *serviceRecord, writer io.Writer) {
	/*
		Example:
			backend backend1
				mode http
				balance leastconn
				server node-1 1.2.3.4:80 check
	*/
	fmt.Fprintf(writer, "backend %s\n", record.serviceName)
	fmt.Fprintf(writer, "    mode http\n")
	fmt.Fprintf(writer, "    balance leastconn\n")
	// strip out the path prefix
	// TODO: should this be optional?
	fmt.Fprintf(writer, "    reqrep ^(GET|POST)\\ %s/(.*) \\1\\ /\\2\n", path)
	for _, subset := range record.endpoints.Subsets {
		// TODO: make this more advanced than just a single port
		port := findPortInSubset(&subset, &record.service.Spec.Ports[0])
		if port != nil {
			for _, address := range subset.Addresses {
				fmt.Fprintf(writer, "    server %s %s:%d check\n", address.TargetRef.Name, address.IP, port.Port)
			}
		}
	}
	fmt.Fprintf(writer, "\n")
}

func (h *HAProxyManager) updateHAProxy() error {
	writer, err := os.OpenFile(h.ConfigFile, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	defer writer.Close()
	h.writeConfig(writer)

	buff := &bytes.Buffer{}
	h.writeConfig(buff)
	glog.Info(buff.String())
	return nil
}

func (h *HAProxyManager) writeConfig(writer io.Writer) {
	io.WriteString(writer, header)
	io.WriteString(writer, "\n")
	for _, serviceRecord := range h.simpleServices {
		service := serviceRecord.service
		endpoints := serviceRecord.endpoints
		for _, servicePort := range service.Spec.Ports {
			writeSimpleService(service, &servicePort, endpoints, "tcp", writer)
		}
	}
	if len(h.routedServices) > 0 {
		h.writeHTTPFrontend(writer)
		for path, serviceRecord := range h.routedServices {
			h.writeHTTPBackend(path, serviceRecord, writer)
		}
	}
}

func (h *HAProxyManager) restartHAProxy() error {
	pid, err := ioutil.ReadFile("/var/run/haproxy.pid")
	if err != nil {
		if os.IsNotExist(err) {
			return h.startHAProxy()
		}
		return err
	}
	return h.runCommandAndLog("haproxy", "-f", h.ConfigFile, "-p", "/var/run/haproxy.pid", "-st", string(pid))
}

func (h *HAProxyManager) startHAProxy() error {
	return h.runCommandAndLog("haproxy", "-f", h.ConfigFile, "-p", "/var/run/haproxy.pid")
}

func (h *HAProxyManager) runCommandAndLog(cmd string, args ...string) error {
	data, err := h.Exec.Command(cmd, args...).CombinedOutput()
	if err != nil {
		glog.Warning("Failed to run: %s %v", cmd, args)
		glog.Warning(string(data))
		return err
	}
	return nil
}

func updateServiceRecord(record *serviceRecord, kubeClient *client.Client) (bool, error) {
	var err error
	record.service, err = kubeClient.Services(record.namespace).Get(record.serviceName)
	if err != nil {
		return false, err
	}
	record.endpoints, err = kubeClient.Endpoints(record.namespace).Get(record.serviceName)
	if err != nil {
		return false, err
	}
	return record.endpoints.ResourceVersion != record.lastVersion, nil
}

func (h *HAProxyManager) AddSimpleService(namespace, service string) {
	h.simpleServices = append(h.simpleServices, serviceRecord{serviceName: service, namespace: namespace})
}

func (h *HAProxyManager) AddRoutedService(path, namespace, service string) {
	if h.routedServices == nil {
		h.routedServices = map[string]*serviceRecord{}
	}
	h.routedServices[path] = &serviceRecord{serviceName: service, namespace: namespace}
}

func (h *HAProxyManager) SyncOnce(kubeClient *client.Client, namespace string) error {
	changes := false
	for ix := range h.simpleServices {
		if updated, err := updateServiceRecord(&h.simpleServices[ix], kubeClient); err != nil {
			return err
		} else if updated {
			changes = true
		}
	}
	for path := range h.routedServices {
		if updated, err := updateServiceRecord(h.routedServices[path], kubeClient); err != nil {
			return err
		} else if updated {
			changes = true
		}
	}
	if changes {
		if err := h.updateHAProxy(); err != nil {
			return err
		}
		if err := h.restartHAProxy(); err != nil {
			return err
		}
		// We've successfully updated haproxy, "commit" the updates to the serviceRecords
		for ix := range h.simpleServices {
			serviceRecord := &h.simpleServices[ix]
			serviceRecord.lastVersion = serviceRecord.endpoints.ResourceVersion
		}
		for _, record := range h.routedServices {
			record.lastVersion = record.endpoints.ResourceVersion
		}
	} else {
		glog.V(2).Infof("Skipping update since version hasn't changed.")
	}
	return nil
}

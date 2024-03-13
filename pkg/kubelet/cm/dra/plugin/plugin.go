/*
Copyright 2022 The Kubernetes Authors.

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

package plugin

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/connectivity"
	"google.golang.org/grpc/credentials/insecure"
	v1 "k8s.io/api/core/v1"
	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/client-go/kubernetes"
	"k8s.io/klog/v2"
)

const (
	// DRAPluginName is the name of the in-tree DRA Plugin.
	DRAPluginName   = "kubernetes.io/dra"
	v1alpha3Version = "v1alpha3"
	v1alpha2Version = "v1alpha2"
)

// Plugin is a description of a DRA Plugin, defined by an endpoint
// and the highest DRA version supported.
type plugin struct {
	sync.Mutex
	conn                    *grpc.ClientConn
	endpoint                string
	version                 string
	highestSupportedVersion *utilversion.Version
	clientTimeout           time.Duration
}

func (p *plugin) getOrCreateGRPCConn() (*grpc.ClientConn, error) {
	p.Lock()
	defer p.Unlock()

	if p.conn != nil {
		return p.conn, nil
	}

	network := "unix"
	klog.V(4).InfoS(log("creating new gRPC connection"), "protocol", network, "endpoint", p.endpoint)
	conn, err := grpc.Dial(
		p.endpoint,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(func(ctx context.Context, target string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, target)
		}),
	)
	if err != nil {
		return nil, err
	}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()

	if ok := conn.WaitForStateChange(ctx, connectivity.Connecting); !ok {
		return nil, errors.New("timed out waiting for gRPC connection to be ready")
	}

	p.conn = conn
	return p.conn, nil
}

func (p *plugin) getVersion() string {
	p.Lock()
	defer p.Unlock()
	return p.version
}

func (p *plugin) setVersion(version string) {
	p.Lock()
	p.version = version
	p.Unlock()
}

// RegistrationHandler is the handler which is fed to the pluginwatcher API.
type RegistrationHandler struct {
	controller *nodeResourcesController
}

// NewPluginHandler returns new registration handler.
//
// Must only be called once per process because it manages global state.
// If a kubeClient is provided, then it synchronizes ResourceSlices
// with the resource information provided by plugins.
func NewRegistrationHandler(kubeClient kubernetes.Interface, getNode func() (*v1.Node, error)) *RegistrationHandler {
	handler := &RegistrationHandler{}

	// If kubelet ever gets an API for stopping registration handlers, then
	// that would need to be hooked up with stopping the controller.
	handler.controller = startNodeResourcesController(context.TODO(), kubeClient, getNode)

	return handler
}

// RegisterPlugin is called when a plugin can be registered.
func (h *RegistrationHandler) RegisterPlugin(pluginName string, endpoint string, versions []string, pluginClientTimeout *time.Duration) error {
	klog.InfoS("Register new DRA plugin", "name", pluginName, "endpoint", endpoint)

	highestSupportedVersion, err := h.validateVersions("RegisterPlugin", pluginName, versions)
	if err != nil {
		return err
	}

	var timeout time.Duration
	if pluginClientTimeout == nil {
		timeout = PluginClientTimeout
	} else {
		timeout = *pluginClientTimeout
	}

	pluginInstance := &plugin{
		conn:                    nil,
		endpoint:                endpoint,
		version:                 v1alpha3Version,
		highestSupportedVersion: highestSupportedVersion,
		clientTimeout:           timeout,
	}

	// Storing endpoint of newly registered DRA Plugin into the map, where plugin name will be the key
	// all other DRA components will be able to get the actual socket of DRA plugins by its name.
	// By default we assume the supported plugin version is v1alpha3
	draPlugins.add(pluginName, pluginInstance)
	h.controller.addPlugin(pluginName, pluginInstance)

	return nil
}

func (h *RegistrationHandler) validateVersions(
	callerName string,
	pluginName string,
	versions []string,
) (*utilversion.Version, error) {
	if len(versions) == 0 {
		return nil, errors.New(
			log(
				"%s for DRA plugin %q failed. Plugin returned an empty list for supported versions",
				callerName,
				pluginName,
			),
		)
	}

	// Validate version
	newPluginHighestVersion, err := utilversion.HighestSupportedVersion(versions)
	if err != nil {
		return nil, errors.New(
			log(
				"%s for DRA plugin %q failed. None of the versions specified %q are supported. err=%v",
				callerName,
				pluginName,
				versions,
				err,
			),
		)
	}

	existingPlugin := draPlugins.get(pluginName)
	if existingPlugin == nil {
		return newPluginHighestVersion, nil
	}
	if existingPlugin.highestSupportedVersion.LessThan(newPluginHighestVersion) {
		return newPluginHighestVersion, nil
	}
	return nil, errors.New(
		log(
			"%s for DRA plugin %q failed. Another plugin with the same name is already registered with a higher supported version: %q",
			callerName,
			pluginName,
			existingPlugin.highestSupportedVersion,
		),
	)
}

func deregisterPlugin(pluginName string) {
	draPlugins.delete(pluginName)
}

// DeRegisterPlugin is called when a plugin has removed its socket,
// signaling it is no longer available.
func (h *RegistrationHandler) DeRegisterPlugin(pluginName string) {
	klog.InfoS("DeRegister DRA plugin", "name", pluginName)
	deregisterPlugin(pluginName)
	h.controller.removePlugin(pluginName)
}

// ValidatePlugin is called by kubelet's plugin watcher upon detection
// of a new registration socket opened by DRA plugin.
func (h *RegistrationHandler) ValidatePlugin(pluginName string, endpoint string, versions []string) error {
	klog.InfoS("Validate DRA plugin", "name", pluginName, "endpoint", endpoint, "versions", strings.Join(versions, ","))

	_, err := h.validateVersions("ValidatePlugin", pluginName, versions)
	if err != nil {
		return fmt.Errorf("validation failed for DRA plugin %s at endpoint %s: %+v", pluginName, endpoint, err)
	}

	return err
}

// log prepends log string with `kubernetes.io/dra`.
func log(msg string, parts ...interface{}) string {
	return fmt.Sprintf(fmt.Sprintf("%s: %s", DRAPluginName, msg), parts...)
}

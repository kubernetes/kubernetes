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

package kubeletplugin

import (
	"context"
	"fmt"
	"sync/atomic"

	registerapi "k8s.io/kubelet/pkg/apis/pluginregistration/v1"
)

// registrationServer implements the kubelet plugin registration gRPC interface.
type registrationServer struct {
	driverName        string
	draEndpointPath   string
	supportedVersions []string
	status            *registerapi.RegistrationStatus

	getInfoError atomic.Pointer[error]

	registerapi.UnsafeRegistrationServer
}

var _ registerapi.RegistrationServer = &registrationServer{}

// GetInfo is the RPC invoked by plugin watcher.
func (e *registrationServer) GetInfo(ctx context.Context, req *registerapi.InfoRequest) (*registerapi.PluginInfo, error) {
	if err := e.getGetInfoError(); err != nil {
		return nil, err
	}
	return &registerapi.PluginInfo{
		Type:              registerapi.DRAPlugin,
		Name:              e.driverName,
		Endpoint:          e.draEndpointPath,
		SupportedVersions: e.supportedVersions,
	}, nil
}

// NotifyRegistrationStatus is the RPC invoked by plugin watcher.
func (e *registrationServer) NotifyRegistrationStatus(ctx context.Context, status *registerapi.RegistrationStatus) (*registerapi.RegistrationStatusResponse, error) {
	e.status = status
	if !status.PluginRegistered {
		return nil, fmt.Errorf("failed registration process: %+v", status.Error)
	}

	return &registerapi.RegistrationStatusResponse{}, nil
}

func (e *registrationServer) getGetInfoError() error {
	errPtr := e.getInfoError.Load()
	if errPtr == nil {
		return nil
	}
	return *errPtr
}

// setGetInfoError sets the error to be returned by the GetInfo handler of the registration server.
// If a non-nil error is provided, subsequent GetInfo calls will return this error.
// Passing nil as the err argument will clear any previously set error, effectively disabling erroring.
func (e *registrationServer) setGetInfoError(err error) {
	if err == nil {
		e.getInfoError.Store(nil)
		return
	}
	e.getInfoError.Store(&err)
}

/*
Copyright 2017 The Kubernetes Authors.

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

package isolator

import (
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
)

type EventDispatcherClient struct {
	Name string
	Ctx  context.Context
	// EventDispatcherClient specified in protobuf
	lifecycle.EventDispatcherClient
}

// Constructor for EventDispatcherClient
func newEventDispatcherClient(name string, serverAddress string) (*EventDispatcherClient, error) {
	clientConn, err := grpc.Dial(serverAddress, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	return &EventDispatcherClient{
		Name: name,
		Ctx:  context.Background(),
		EventDispatcherClient: lifecycle.NewEventDispatcherClient(clientConn),
	}, nil
}

// Create RegisterRequest and register EventDispatcherClient
func (e *EventDispatcherClient) RegisterIsolator(clientAddress string) error {
	registerRequest := &lifecycle.RegisterRequest{
		SocketAddress: clientAddress,
		Name:          e.Name,
	}

	if _, err := e.EventDispatcherClient.Register(e.Ctx, registerRequest); err != nil {
		return err
	}

	return nil
}

func (e *EventDispatcherClient) UnregisterIsolator() error {
	_, err := e.Unregister(e.Ctx, &lifecycle.UnregisterRequest{
		Name: e.Name,
	})
	return err
}

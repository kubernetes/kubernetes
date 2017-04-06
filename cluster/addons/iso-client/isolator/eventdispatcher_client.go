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
func newEventDispatcherClient(serverAddress string, name string) (*EventDispatcherClient, error) {
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
func RegisterIsolator(name string, serverAddress string, clientAddress string) (*EventDispatcherClient, error) {
	client, err := newEventDispatcherClient(serverAddress, name)
	if err != nil {
		return nil, err
	}
	registerRequest := &lifecycle.RegisterRequest{
		SocketAddress: clientAddress,
		Name:          client.Name,
	}

	if _, err := client.Register(client.Ctx, registerRequest); err != nil {
		return nil, err
	}

	return client, nil
}

func (e *EventDispatcherClient) UnregisterIsolator() error {
	_, err := e.Unregister(e.Ctx, &lifecycle.UnregisterRequest{
		Name: e.Name,
	})
	return err
}

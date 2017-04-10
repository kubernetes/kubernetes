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
func NewEventDispatcherClient(name string, serverAddress string) (*EventDispatcherClient, error) {
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

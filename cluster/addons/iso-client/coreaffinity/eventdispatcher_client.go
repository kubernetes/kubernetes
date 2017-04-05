package coreaffinity

import (
	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
)

type EventDispatcherClient struct {
	Name    string
	Address string
	Ctx     context.Context
	// EventDispatcherClient specified in protobuf
	lifecycle.EventDispatcherClient
}

// Constructor for EventDispatcherClient
func NewEventDispatcherClient(name string, serverAddress string, clientAddress string) (*EventDispatcherClient, error) {
	clientConn, err := grpc.Dial(serverAddress, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	return &EventDispatcherClient{
			EventDispatcherClient: lifecycle.NewEventDispatcherClient(clientConn),
			Ctx:     context.Background(),
			Name:    name,
			Address: clientAddress,
		},
		nil
}

// Create RegisterRequest and register EventDispatcherClient
func (edc *EventDispatcherClient) Register() (reply *lifecycle.RegisterReply, err error) {
	registerRequest := &lifecycle.RegisterRequest{
		SocketAddress: edc.Address,
		Name:          edc.Name,
	}

	glog.Infof("Attempting to register EventDispatcherClient. Request: %v", registerRequest)
	reply, err = edc.EventDispatcherClient.Register(edc.Ctx, registerRequest)
	if err != nil {
		return reply, err
	}
	return reply, nil
}

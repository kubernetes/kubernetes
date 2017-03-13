package coreaffinity

import (
	"os"

	"github.com/golang/glog"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
	"k8s.io/kubernetes/pkg/util/uuid"
)

type eventDispatcherClient struct {
	Token   string
	Name    string
	Address string
	Ctx     context.Context
	// EventDispatcherClient specified in protobuf
	lifecycle.EventDispatcherClient
}

// Constructor for eventDispatcherClient
func NewEventDispatcherClient(name string, serverAddress string, clientAddress string) (*eventDispatcherClient, error) {
	clientConn, err := grpc.Dial(serverAddress, grpc.WithInsecure())
	if err != nil {
		return nil, err
	}
	return &eventDispatcherClient{
			EventDispatcherClient: lifecycle.NewEventDispatcherClient(clientConn),
			Ctx:     context.Background(),
			Name:    name,
			Address: clientAddress,
		},
		nil

}

// Create RegisterRequest and register eventDispatcherClient
func (edc *eventDispatcherClient) Register() (reply *lifecycle.RegisterReply, err error) {
	registerToken := string(uuid.NewUUID())
	registerRequest := &lifecycle.RegisterRequest{
		SocketAddress: edc.Address,
		Name:          edc.Name,
		Token:         registerToken,
	}

	glog.Infof("Attempting to register evenDispatcherClient. Request: %v", registerRequest)
	reply, err = edc.EventDispatcherClient.Register(edc.Ctx, registerRequest)
	if err != nil {
		return reply, err
	}
	edc.Token = reply.Token
	return reply, nil
}

// TODO: handle more than just unregistering  evenDispatcherClient
func HandleSIGTERM(c chan os.Signal, client *eventDispatcherClient) {
	<-c
	unregisterRequest := &lifecycle.UnregisterRequest{
		Name:  client.Name,
		Token: client.Token,
	}
	rep, err := client.Unregister(client.Ctx, unregisterRequest)
	if err != nil {
		glog.Fatalf("Failed to unregister handler: %v")
		os.Exit(1)
	}
	glog.Infof("Unregistering iso: %v\n", rep)

	os.Exit(0)

}

package flexvolume

import (
	"time"
	
	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes/empty"
	"go.pedge.io/dlog"
	"golang.org/x/net/context"
)

type apiServer struct {
	client Client
}

func newAPIServer(client Client) *apiServer {
	return &apiServer{client}
}

func (a *apiServer) Init(_ context.Context, _ *empty.Empty) (_ *empty.Empty, err error) {
	defer func(start time.Time) { log("Init", nil, nil, err, time.Since(start)) }(time.Now())
	return checkClientError(a.client.Init())
}

func (a *apiServer) Attach(_ context.Context, request *AttachRequest) (_ *empty.Empty, err error) {
	defer func(start time.Time) { log("Attach", request, nil, err, time.Since(start)) }(time.Now())
	return checkClientError(a.client.Attach(request.JsonOptions))
}

func (a *apiServer) Detach(_ context.Context, request *DetachRequest) (_ *empty.Empty, err error) {
	defer func(start time.Time) { log("Detach", request, nil, err, time.Since(start)) }(time.Now())
	return checkClientError(a.client.Detach(request.MountDevice, false))
}

func (a *apiServer) Mount(_ context.Context, request *MountRequest) (_ *empty.Empty, err error) {
	defer func(start time.Time) { log("Mount", request, nil, err, time.Since(start)) }(time.Now())
	return checkClientError(a.client.Mount(request.TargetMountDir, request.MountDevice, request.JsonOptions))
}

func (a *apiServer) Unmount(_ context.Context, request *UnmountRequest) (_ *empty.Empty, err error) {
	defer func(start time.Time) { log("Unmount", request, nil, err, time.Since(start)) }(time.Now())
	return checkClientError(a.client.Unmount(request.MountDir))
}

func checkClientError(err error) (*empty.Empty, error) {
	if err != nil {
		return nil, err
	}
	return &empty.Empty{}, nil
}

func log(methodName string, request proto.Message, response proto.Message, err error, duration time.Duration) {
	if err != nil {
		dlog.Errorf("Method: %v Request: %v Response: %v Error: %v Duration: %v",
			methodName, request.String(), response.String(), err.Error(), duration.String())
	} else {
		dlog.Infof("Method: %v Request: %v Response: %v Error: %v Duration: %v",
			methodName, request.String(), response.String(), err.Error(), duration.String())
	}
}

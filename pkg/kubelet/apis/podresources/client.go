/*
Copyright 2018 The Kubernetes Authors.

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

package podresources

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	"k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubelet/pkg/apis/podresources/v1alpha1"
	"k8s.io/kubernetes/pkg/kubelet/util"
)

// Note: Consumers of the pod resources API should not be importing this package.
// They should copy paste the function in their project.

const (
	DefaultTimeout    = 10 * time.Second
	DefaultMaxMsgSize = 1024 * 1024 * 16 // 16 MiB
)

// GetV1alpha1Client returns a client for the PodResourcesLister grpc service
// Note: This is deprecated
func GetV1alpha1Client(socket string, connectionTimeout time.Duration, maxMsgSize int) (v1alpha1.PodResourcesListerClient, *grpc.ClientConn, error) {
	addr, dialer, err := util.GetAddressAndDialer(socket)
	if err != nil {
		return nil, nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), connectionTimeout)
	defer cancel()

	conn, err := grpc.DialContext(ctx, addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(dialer),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxMsgSize)))
	if err != nil {
		return nil, nil, fmt.Errorf("error dialing socket %s: %v", socket, err)
	}
	return v1alpha1.NewPodResourcesListerClient(conn), conn, nil
}

// GetV1Client returns a client for the PodResourcesLister grpc service
func GetV1Client(socket string, connectionTimeout time.Duration, maxMsgSize int) (v1.PodResourcesListerClient, *grpc.ClientConn, error) {
	addr, dialer, err := util.GetAddressAndDialer(socket)
	if err != nil {
		return nil, nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), connectionTimeout)
	defer cancel()

	conn, err := grpc.DialContext(ctx, addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(dialer),
		grpc.WithDefaultCallOptions(grpc.MaxCallRecvMsgSize(maxMsgSize)))
	if err != nil {
		return nil, nil, fmt.Errorf("error dialing socket %s: %v", socket, err)
	}
	return v1.NewPodResourcesListerClient(conn), conn, nil
}

// GetClient returns a client for the recommended version of the PodResourcesLister grpc service with the recommended settings
func GetClient(endpoint string) (v1.PodResourcesListerClient, *grpc.ClientConn, error) {
	return GetV1Client(endpoint, DefaultTimeout, DefaultMaxMsgSize)
}

// WaitForReady ensures the communication has been established.
// We provide a composable WaitForReady instead of setting flags in the Dialing function to enable client code flexibility.
// In general, using `grpc.Dial` with the blocking flag enabled is an anti-pattern https://github.com/grpc/grpc-go/blob/master/Documentation/anti-patterns.md
// But things are a bit different in the very narrow case we use here, over local UNIX domain socket. The transport is very stable and lossless,
// and the most common cause for failures bubbling up is for kubelet not yet ready, which is very common in the e2e tests but much less
// in the expected normal operation.
func WaitForReady(cli v1.PodResourcesListerClient, conn *grpc.ClientConn, err error) (v1.PodResourcesListerClient, *grpc.ClientConn, error) {
	if err != nil {
		return cli, conn, err
	}
	// we use List because it's the oldest endpoint and the one guaranteed to be available.
	// Note we only set WaitForReady explicitly here effectively triggering eager connection. This way we force the connection to happen
	// (or fail critically) without forcing the client code to use `grpc.WaitForReady` in their code everywhere.
	// TODO: evaluate more lightweight option like GetAllocatableResources - we will discard the return value anyway.
	_, listErr := cli.List(context.Background(), &v1.ListPodResourcesRequest{}, grpc.WaitForReady(true))
	if listErr != nil {
		return cli, conn, fmt.Errorf("WaitForReady failed: %w", listErr)
	}
	return cli, conn, nil
}

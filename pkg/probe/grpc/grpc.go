/*
Copyright 2015 The Kubernetes Authors.

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

package grpc

import (
	"context"
	"errors"
	"fmt"
	"google.golang.org/grpc"
	hcservice "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/metadata"
	"k8s.io/component-base/version"
	"k8s.io/kubernetes/pkg/probe"
	"time"
)

var (
	errGrpcNotServing = errors.New("GRPC_STATUS_NOT_SERVING")
)

// Prober is an interface that defines the Probe function for doing GRPC readiness/liveness checks.
type Prober interface {
	Probe(host string, port int, timeout time.Duration, opts ...grpc.DialOption) (probe.Result, string, error)
}

type grpcProber struct {
}

// New Prober for execute grpc probe test
func New() Prober {
	return grpcProber{}
}

// Probe executes a grpc call to check the liveness/readiness of container.
// Returns the Result status, command output, and errors if any.
func (p grpcProber) Probe(host string, port int, timeout time.Duration, opts ...grpc.DialOption) (probe.Result, string, error) {
	v := version.Get()

	md := metadata.New(map[string]string{
		"User-Agent": fmt.Sprintf("kube-probe/%s.%s", v.Major, v.Minor),
	})

	ctx, cancel := context.WithTimeout(context.Background(), timeout)

	defer cancel()

	conn, err := grpc.DialContext(ctx, fmt.Sprintf("%s:%d", host, port), opts...)

	if err != nil {
		return probe.Failure, fmt.Sprintf("GRPC probe failed with error: %s", err.Error()), err
	}

	defer func() {
		_ = conn.Close()
	}()

	client := hcservice.NewHealthClient(conn)

	res, err := client.Check(metadata.NewOutgoingContext(ctx, md), &hcservice.HealthCheckRequest{})

	if err != nil {
		return probe.Failure, fmt.Sprintf("GRPC probe failed with error: %s", err.Error()), err
	}

	if res.Status != hcservice.HealthCheckResponse_SERVING {
		return probe.Failure, fmt.Sprintf("GRPC probe failed with status: %s", res.Status.String()), errGrpcNotServing
	}

	return probe.Success, fmt.Sprintf("GRPC probe success"), nil
}

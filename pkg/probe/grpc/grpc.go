/*
Copyright 2021 The Kubernetes Authors.

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
	"fmt"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	grpchealth "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"k8s.io/component-base/version"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/probe"
)

// Prober is an interface that defines the Probe function for doing GRPC readiness/liveness/startup checks.
type Prober interface {
	Probe(host, service string, port int, timeout time.Duration, opts ...grpc.DialOption) (probe.Result, string, error)
}

type grpcProber struct {
}

// New Prober for execute grpc probe
func New() Prober {
	return grpcProber{}
}

// Probe executes a grpc call to check the liveness/readiness/startup of container.
// Returns the Result status, command output, and errors if any.
// Only return non-nil error when service is unavailable and/or not implementing the interface,
// otherwise result status is failed,BUT err is nil
func (p grpcProber) Probe(host, service string, port int, timeout time.Duration, opts ...grpc.DialOption) (probe.Result, string, error) {
	v := version.Get()

	md := metadata.New(map[string]string{
		"User-Agent": fmt.Sprintf("kube-probe/%s.%s", v.Major, v.Minor),
	})

	ctx, cancel := context.WithTimeout(context.Background(), timeout)

	defer cancel()

	addr := net.JoinHostPort(host, fmt.Sprintf("%d", port))
	conn, err := grpc.DialContext(ctx, addr, opts...)

	if err != nil {
		if err == context.DeadlineExceeded {
			klog.V(4).ErrorS(err, "failed to connect grpc service due to timeout", "addr", addr, "service", service, "timeout", timeout)
			return probe.Failure, fmt.Sprintf("GRPC probe failed to dial: %s", err), nil
		} else {
			klog.V(4).ErrorS(err, "failed to connect grpc service", "service", addr)
			return probe.Failure, "", fmt.Errorf("GRPC probe failed to dial: %w", err)
		}
	}

	defer func() {
		_ = conn.Close()
	}()

	client := grpchealth.NewHealthClient(conn)

	resp, err := client.Check(metadata.NewOutgoingContext(ctx, md), &grpchealth.HealthCheckRequest{
		Service: service,
	})

	if err != nil {
		state, ok := status.FromError(err)
		if ok {
			switch state.Code() {
			case codes.Unimplemented:
				klog.V(4).ErrorS(err, "server does not implement the grpc health protocol (grpc.health.v1.Health)", "addr", addr, "service", service)
				return probe.Failure, "", fmt.Errorf("server does not implement the grpc health protocol: %w", err)
			case codes.DeadlineExceeded:
				klog.V(4).ErrorS(err, "rpc request not finished within timeout", "addr", addr, "service", service, "timeout", timeout)
				return probe.Failure, fmt.Sprintf("GRPC probe failed with DeadlineExceeded"), nil
			default:
				klog.V(4).ErrorS(err, "rpc probe failed")
			}
		} else {
			klog.V(4).ErrorS(err, "health rpc probe failed")
		}

		return probe.Failure, "", fmt.Errorf("health rpc probe failed: %w", err)
	}

	if resp.Status != grpchealth.HealthCheckResponse_SERVING {
		return probe.Failure, fmt.Sprintf("GRPC probe failed with status: %s", resp.Status.String()), nil
	}

	return probe.Success, fmt.Sprintf("GRPC probe success"), nil
}

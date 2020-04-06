package grpc

import (
	"context"
	"errors"
	"fmt"
	"google.golang.org/grpc"
	health_check "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/metadata"
	"k8s.io/component-base/version"
	"k8s.io/kubernetes/pkg/probe"
	"time"
)

var (
	errGrpcNotServing = errors.New("GRPC_STATUS_NOT_SERVING")
)

type Prober interface {
	Probe(host string, port int, timeout time.Duration, opts ...grpc.DialOption) (probe.Result, string, error)
}

type grpcProber struct {
}

// Create new Prober for execute grpc probe test
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

	client := health_check.NewHealthClient(conn)

	res, err := client.Check(metadata.NewOutgoingContext(ctx, md), &health_check.HealthCheckRequest{})

	if err != nil {
		return probe.Failure, fmt.Sprintf("GRPC probe failed with error: %s", err.Error()), err
	}

	if res.Status != health_check.HealthCheckResponse_SERVING {
		return probe.Failure, fmt.Sprintf("GRPC probe failed with status: %s", res.Status.String()), errGrpcNotServing
	}

	return probe.Success, fmt.Sprintf("GRPC probe success"), nil
}

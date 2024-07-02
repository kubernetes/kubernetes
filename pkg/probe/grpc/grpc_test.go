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
	"net/http"
	"net/http/httptest"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	grpchealth "google.golang.org/grpc/health/grpc_health_v1"

	"k8s.io/kubernetes/pkg/probe"
)

func TestNew(t *testing.T) {
	t.Run("Should: implement Probe interface", func(t *testing.T) {
		s := New()
		assert.Implements(t, (*Prober)(nil), s)
	})
}

type successServerMock struct {
}

func (s successServerMock) Check(context.Context, *grpchealth.HealthCheckRequest) (*grpchealth.HealthCheckResponse, error) {
	return &grpchealth.HealthCheckResponse{
		Status: grpchealth.HealthCheckResponse_SERVING,
	}, nil
}

func (s successServerMock) Watch(_ *grpchealth.HealthCheckRequest, stream grpchealth.Health_WatchServer) error {
	return stream.Send(&grpchealth.HealthCheckResponse{
		Status: grpchealth.HealthCheckResponse_SERVING,
	})
}

type errorTimeoutServerMock struct {
}

func (e errorTimeoutServerMock) Check(context.Context, *grpchealth.HealthCheckRequest) (*grpchealth.HealthCheckResponse, error) {
	time.Sleep(time.Second * 4)
	return &grpchealth.HealthCheckResponse{
		Status: grpchealth.HealthCheckResponse_SERVING,
	}, nil
}

func (e errorTimeoutServerMock) Watch(_ *grpchealth.HealthCheckRequest, stream grpchealth.Health_WatchServer) error {
	time.Sleep(time.Second * 4)
	return stream.Send(&grpchealth.HealthCheckResponse{
		Status: grpchealth.HealthCheckResponse_SERVING,
	})
}

type errorNotServeServerMock struct {
}

func (e errorNotServeServerMock) Check(context.Context, *grpchealth.HealthCheckRequest) (*grpchealth.HealthCheckResponse, error) {
	return &grpchealth.HealthCheckResponse{
		Status: grpchealth.HealthCheckResponse_NOT_SERVING,
	}, nil
}

func (e errorNotServeServerMock) Watch(_ *grpchealth.HealthCheckRequest, stream grpchealth.Health_WatchServer) error {
	return stream.Send(&grpchealth.HealthCheckResponse{
		Status: grpchealth.HealthCheckResponse_NOT_SERVING,
	})
}

func TestGrpcProber_Probe(t *testing.T) {
	t.Run("Should: failed but return nil error because cant find host", func(t *testing.T) {
		s := New()
		p, o, err := s.Probe("", "", 32, time.Second)
		assert.Equal(t, probe.Failure, p)
		assert.Equal(t, nil, err)
		assert.Equal(t, "timeout: failed to connect service \":32\" within 1s: context deadline exceeded", o)
	})
	t.Run("Should: return nil error because connection closed", func(t *testing.T) {
		s := New()
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			fmt.Fprint(w, "res")
		}))
		u := strings.Split(server.URL, ":")
		assert.Len(t, u, 3)

		port, err := strconv.Atoi(u[2])
		assert.Equal(t, nil, err)

		// take some time to wait server boot
		time.Sleep(2 * time.Second)
		p, _, err := s.Probe("127.0.0.1", "", port, time.Second)
		assert.Equal(t, probe.Failure, p)
		assert.Equal(t, nil, err)
	})
	t.Run("Should: return nil error because server response not served", func(t *testing.T) {
		s := New()
		lis, _ := net.Listen("tcp", ":0")
		port := lis.Addr().(*net.TCPAddr).Port
		grpcServer := grpc.NewServer()
		defer grpcServer.Stop()
		grpchealth.RegisterHealthServer(grpcServer, &errorNotServeServerMock{})
		go func() {
			_ = grpcServer.Serve(lis)
		}()
		// take some time to wait server boot
		time.Sleep(2 * time.Second)
		p, o, err := s.Probe("0.0.0.0", "", port, time.Second)
		assert.Equal(t, probe.Failure, p)
		assert.Equal(t, nil, err)
		assert.Equal(t, "service unhealthy (responded with \"NOT_SERVING\")", o)
	})
	t.Run("Should: return nil-error because server not response in time", func(t *testing.T) {
		s := New()
		lis, _ := net.Listen("tcp", ":0")
		port := lis.Addr().(*net.TCPAddr).Port

		grpcServer := grpc.NewServer()
		defer grpcServer.Stop()
		grpchealth.RegisterHealthServer(grpcServer, &errorTimeoutServerMock{})
		go func() {
			_ = grpcServer.Serve(lis)
		}()
		// take some time to wait server boot
		time.Sleep(2 * time.Second)
		p, o, err := s.Probe("0.0.0.0", "", port, time.Second*2)
		assert.Equal(t, probe.Failure, p)
		assert.Equal(t, nil, err)
		assert.Equal(t, "timeout: health rpc did not complete within 2s", o)

	})
	t.Run("Should: not return error because check was success", func(t *testing.T) {
		s := New()
		lis, _ := net.Listen("tcp", ":0")
		port := lis.Addr().(*net.TCPAddr).Port

		grpcServer := grpc.NewServer()
		defer grpcServer.Stop()
		grpchealth.RegisterHealthServer(grpcServer, &successServerMock{})
		go func() {
			_ = grpcServer.Serve(lis)
		}()
		// take some time to wait server boot
		time.Sleep(2 * time.Second)
		p, _, err := s.Probe("0.0.0.0", "", port, time.Second*2)
		assert.Equal(t, probe.Success, p)
		assert.Equal(t, nil, err)
	})
	t.Run("Should: not return error because check was success, when listen port is 0", func(t *testing.T) {
		s := New()
		lis, _ := net.Listen("tcp", ":0")
		port := lis.Addr().(*net.TCPAddr).Port

		grpcServer := grpc.NewServer()
		defer grpcServer.Stop()
		grpchealth.RegisterHealthServer(grpcServer, &successServerMock{})
		go func() {
			_ = grpcServer.Serve(lis)
		}()
		// take some time to wait server boot
		time.Sleep(2 * time.Second)
		p, _, err := s.Probe("0.0.0.0", "", port, time.Second*2)
		assert.Equal(t, probe.Success, p)
		assert.Equal(t, nil, err)
	})
}

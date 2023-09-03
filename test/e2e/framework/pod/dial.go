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

package pod

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"regexp"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/client-go/transport/spdy"
	"k8s.io/klog/v2"
)

// NewTransport creates a transport which uses the port forward dialer.
// URLs must use <namespace>.<pod>:<port> as host.
func NewTransport(client kubernetes.Interface, restConfig *rest.Config) *http.Transport {
	return &http.Transport{
		DialContext: func(ctx context.Context, _, addr string) (net.Conn, error) {
			dialer := NewDialer(client, restConfig)
			a, err := ParseAddr(addr)
			if err != nil {
				return nil, err
			}
			return dialer.DialContainerPort(ctx, *a)
		},
	}
}

// NewDialer creates a dialer that supports connecting to container ports.
func NewDialer(client kubernetes.Interface, restConfig *rest.Config) *Dialer {
	return &Dialer{
		client:     client,
		restConfig: restConfig,
	}
}

// Dialer holds the relevant parameters that are independent of a particular connection.
type Dialer struct {
	client     kubernetes.Interface
	restConfig *rest.Config
}

// DialContainerPort connects to a certain container port in a pod.
func (d *Dialer) DialContainerPort(ctx context.Context, addr Addr) (conn net.Conn, finalErr error) {
	restClient := d.client.CoreV1().RESTClient()
	restConfig := d.restConfig
	if restConfig.GroupVersion == nil {
		restConfig.GroupVersion = &schema.GroupVersion{}
	}
	if restConfig.NegotiatedSerializer == nil {
		restConfig.NegotiatedSerializer = scheme.Codecs
	}

	// The setup code around the actual portforward is from
	// https://github.com/kubernetes/kubernetes/blob/c652ffbe4a29143623a1aaec39f745575f7e43ad/staging/src/k8s.io/kubectl/pkg/cmd/portforward/portforward.go
	req := restClient.Post().
		Resource("pods").
		Namespace(addr.Namespace).
		Name(addr.PodName).
		SubResource("portforward")
	transport, upgrader, err := spdy.RoundTripperFor(restConfig)
	if err != nil {
		return nil, fmt.Errorf("create round tripper: %w", err)
	}
	dialer := spdy.NewDialer(upgrader, &http.Client{Transport: transport}, "POST", req.URL())

	streamConn, _, err := dialer.Dial(portforward.PortForwardProtocolV1Name)
	if err != nil {
		return nil, fmt.Errorf("dialer failed: %w", err)
	}
	requestID := "1"
	defer func() {
		if finalErr != nil {
			streamConn.Close()
		}
	}()

	// create error stream
	headers := http.Header{}
	headers.Set(v1.StreamType, v1.StreamTypeError)
	headers.Set(v1.PortHeader, fmt.Sprintf("%d", addr.Port))
	headers.Set(v1.PortForwardRequestIDHeader, requestID)

	// We're not writing to this stream, just reading an error message from it.
	// This happens asynchronously.
	errorStream, err := streamConn.CreateStream(headers)
	if err != nil {
		return nil, fmt.Errorf("error creating error stream: %w", err)
	}
	errorStream.Close()
	go func() {
		message, err := io.ReadAll(errorStream)
		switch {
		case err != nil:
			klog.ErrorS(err, "error reading from error stream")
		case len(message) > 0:
			klog.ErrorS(errors.New(string(message)), "an error occurred connecting to the remote port")
		}
	}()

	// create data stream
	headers.Set(v1.StreamType, v1.StreamTypeData)
	dataStream, err := streamConn.CreateStream(headers)
	if err != nil {
		return nil, fmt.Errorf("error creating data stream: %w", err)
	}

	return &stream{
		Stream:     dataStream,
		streamConn: streamConn,
	}, nil
}

// Addr contains all relevant parameters for a certain port in a pod.
// The container should be running before connections are attempted,
// otherwise the connection will fail.
type Addr struct {
	Namespace, PodName string
	Port               int
}

var _ net.Addr = Addr{}

func (a Addr) Network() string {
	return "port-forwarding"
}

func (a Addr) String() string {
	return fmt.Sprintf("%s.%s:%d", a.Namespace, a.PodName, a.Port)
}

// ParseAddr expects a <namespace>.<pod>:<port number> as produced
// by Addr.String.
func ParseAddr(addr string) (*Addr, error) {
	parts := addrRegex.FindStringSubmatch(addr)
	if parts == nil {
		return nil, fmt.Errorf("%q: must match the format <namespace>.<pod>:<port number>", addr)
	}
	port, _ := strconv.Atoi(parts[3])
	return &Addr{
		Namespace: parts[1],
		PodName:   parts[2],
		Port:      port,
	}, nil
}

var addrRegex = regexp.MustCompile(`^([^\.]+)\.([^:]+):(\d+)$`)

type stream struct {
	addr Addr
	httpstream.Stream
	streamConn httpstream.Connection
}

var _ net.Conn = &stream{}

func (s *stream) Close() error {
	s.Stream.Close()
	s.streamConn.Close()
	return nil
}

func (s *stream) LocalAddr() net.Addr {
	return LocalAddr{}
}

func (s *stream) RemoteAddr() net.Addr {
	return s.addr
}

func (s *stream) SetDeadline(t time.Time) error {
	return nil
}

func (s *stream) SetReadDeadline(t time.Time) error {
	return nil
}

func (s *stream) SetWriteDeadline(t time.Time) error {
	return nil
}

type LocalAddr struct{}

var _ net.Addr = LocalAddr{}

func (l LocalAddr) Network() string { return "port-forwarding" }
func (l LocalAddr) String() string  { return "apiserver" }

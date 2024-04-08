/*
Copyright 2024 The Kubernetes Authors.

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

package restproxy

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"sync"
	"time"

	"google.golang.org/grpc"

	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	restproxyapi "k8s.io/dynamic-resource-allocation/apis/restproxy/v1alpha1"
	"k8s.io/klog/v2"
)

// readChunkSize is the maximum amount of data sent per gRPC request.
//
// There is a 4MB size limit by default for gRPC messages, which thankfully
// is smaller than the maximum size of requests to the apiserver (https://github.com/kubernetes/apiserver/blame/8d18eec7c050338aac4d49e470f3ea0b946f4726/pkg/server/config.go#L442).
// Therefore we can be sure that any request to the apiserver
// fits into one gRPC message and that we only need to split the response
// data.
const readChunkSize = 3 * 1024 * 1024

// RequestFilter can be used to modify requests or reject them.
type RequestFilter interface {
	// FilterRequest may modify the request in place. If the request should
	// not be executed, an error is returned.
	//
	// The request is the one that is going to be sent to the API server.
	// The path there may have additional entries if the API is not located
	// at the root. The separate apiPath is the one without such a prefix.
	FilterRequest(ctx context.Context, req *http.Request, apiPath string) error
}

// StartRESTProxy connects to the proxy client and handles its REST requests.
//
// It automatically retries after errors, including "not implemented" errors
// because even those might be temporary (the gRPC connection might
// get re-established after restarting the server with a newer implementation
// which then implements it).
//
// Canceling the context stops all background activity. [WaitForShutdown] can
// be used block until all goroutines have stopped after cancellation.
func StartRESTProxy(ctx context.Context, baseURL *url.URL, httpClient *http.Client, grpcConn *grpc.ClientConn, filter RequestFilter) *RESTProxy {
	p := &RESTProxy{
		baseURL:    baseURL,
		httpClient: httpClient,
		restClient: restproxyapi.NewRESTClient(grpcConn),
		filter:     filter,
	}

	logger := klog.FromContext(ctx)
	ctx, p.cancel = context.WithCancelCause(ctx)
	logger.V(3).Info("Starting")
	p.wg.Add(1)
	go func() {
		defer p.wg.Done()
		defer logger.V(3).Info("Stopping")
		defer p.cancel(errors.New("RESTProxy stopped"))
		defer utilruntime.HandleCrash()
		p.run(ctx)
	}()

	return p
}

// RESTProxy is receiving REST requests from a gRPC server in a stream,
// forwards to the apiserver, and returns the response.
type RESTProxy struct {
	cancel     func(cause error)
	baseURL    *url.URL
	httpClient *http.Client
	restClient restproxyapi.RESTClient
	filter     RequestFilter
	wg         sync.WaitGroup
}

// Stop causes all goroutines to stop immediately and waits for them to stop.
// Canceling the context passed to Start also causes goroutines to stop.
func (p *RESTProxy) Stop() {
	p.cancel(errors.New("RESTProxy stopping"))
	p.wg.Wait()
}

func (p *RESTProxy) run(ctx context.Context) {
	logger := klog.FromContext(ctx)
	for ctx.Err() == nil {
		backoff := wait.Backoff{
			Duration: time.Second,
			Factor:   2,
			Jitter:   0.2,
			Cap:      time.Minute,
			Steps:    100,
		}
		_ = wait.ExponentialBackoffWithContext(ctx, backoff, func(ctx context.Context) (bool, error) {
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			proxyClient, err := p.restClient.Proxy(ctx, &restproxyapi.ProxyMessage{})
			if err != nil {
				logger.V(5).Info("Failed to connect", "err", err)
				// Retry with exponential backoff.
				return false, nil
			}

			err = p.handleClient(ctx, proxyClient, cancel)
			logger.V(3).Info("Client has been handled", "err", err)
			if err != nil {
				// Retry with exponential backoff.
				return false, nil
			}
			// Return from ExponentialBackoffWithContext and try to connect anew.
			return true, nil
		})
	}
}

// handleClient handles REST requests, without retrying.
// An error is returned if the proxy client has failed.
func (p *RESTProxy) handleClient(ctx context.Context, proxyClient restproxyapi.REST_ProxyClient, cancelClient func()) error {
	logger := klog.FromContext(ctx)
	for ctx.Err() == nil {
		request, err := proxyClient.Recv()
		if err != nil {
			if errors.Is(err, io.EOF) {
				// Normal shutdown, reconnect with new backoff.
				return nil
			}
			return fmt.Errorf("receiving REST request: %v", err)
		}

		p.wg.Add(1)
		go func() {
			defer p.wg.Done()
			defer utilruntime.HandleCrash()
			// Dump request with most fields.
			logger.V(5).Info("Handling request", "request", logRequest{logger: &logger, request: request})
			// Now include only the ID going forward.
			logger := klog.LoggerWithValues(logger, "request", logRequest{request: request})
			ctx := klog.NewContext(ctx, logger)
			err := p.handleRequest(ctx, request)
			logger.V(5).Info("Handled request", "err", err)
			if err != nil {
				// Abort serving this client, something went wrong.
				cancelClient()
			}
		}()
	}
	return nil
}

// handleRequest handles one REST requests, without retrying.
// An error is returned if the proxy client has failed.
func (p *RESTProxy) handleRequest(ctx context.Context, request *restproxyapi.Request) (finalErr error) {
	logger := klog.FromContext(ctx)
	body := bytes.NewReader(request.Body)
	url := p.baseURL
	url = url.JoinPath(request.Path)
	httpRequest, err := http.NewRequestWithContext(ctx, request.Method, url.String(), body)
	httpRequest.URL.RawQuery = request.RawQuery
	if err != nil {
		return p.sendError(ctx, request, "create HTTP request", err)
	}
	for key, values := range request.Header {
		for _, value := range values.Values {
			httpRequest.Header.Add(key, value)
		}
	}

	// Intercept and/or modify outgoing requests.
	if p.filter != nil {
		if err := p.filter.FilterRequest(ctx, httpRequest, request.Path); err != nil {
			return p.sendError(ctx, request, "filter request", err)
		}
	}

	response, err := p.httpClient.Do(httpRequest)
	if err != nil {
		return p.sendError(ctx, request, "do HTTP request", err)
	}
	defer func() {
		// Ignore follow-up errors, only return it to the caller if it's the only error.
		if err := response.Body.Close(); err != nil && finalErr != nil {
			finalErr = fmt.Errorf("close response body: %v", err)
		}
	}()

	logger.V(5).Info("Starting to forward response", "response", logResponse{logger: &logger, response: response})
	logger = klog.LoggerWithValues(logger, "response", logResponse{response: response})
	ctx = klog.NewContext(ctx, logger)

	buffer := make([]byte, readChunkSize)
	bodyOffset := int64(0)
	for ctx.Err() == nil {
		read, err := response.Body.Read(buffer)
		eof := false
		if errors.Is(err, io.EOF) {
			eof = true
			err = nil
		}
		if err != nil {
			return p.sendError(ctx, request, "read HTTP response body", err)
		}
		reply := &restproxyapi.ReplyMessage{
			Id:         request.Id,
			BodyOffset: bodyOffset,
			Body:       buffer[0:read],
			Header: &restproxyapi.ResponseHeader{
				Status:        response.Status,
				StatusCode:    int32(response.StatusCode),
				Proto:         response.Proto,
				ProtoMajor:    int32(response.ProtoMajor),
				ProtoMinor:    int32(response.ProtoMinor),
				ContentLength: response.ContentLength,
				Header:        make(map[string]*restproxyapi.RESTHeader),
			},
			Close: eof,
		}
		for key, values := range response.Header {
			reply.Header.Header[key] = &restproxyapi.RESTHeader{Values: values}
		}

		logger := klog.LoggerWithValues(logger, "bodyOffset", bodyOffset, "bodyRead", read)
		ctx := klog.NewContext(ctx, logger)
		bodyOffset += int64(read)

		logger.V(5).Info("Body reply")
		replyResponse, err := p.restClient.Reply(ctx, reply)
		if err != nil {
			logger.V(5).Info("Failed to send reply", "clientErr", err)
			return fmt.Errorf("send response body chunk: %v", err)
		}
		if replyResponse.GetClose() {
			logger.V(5).Info("Closing response body as instructed by client")
			// defer above does the response.Body.Close, no need to replicate
			// that here.
			break
		}
		if eof {
			break
		}
	}
	logger.V(5).Info("Forwarded response", "bodyOffset", bodyOffset)

	return nil
}

func (p *RESTProxy) sendError(ctx context.Context, request *restproxyapi.Request, what string, err error) error {
	if cause := context.Cause(ctx); cause != nil {
		// Context is canceled, provide the details better than gRPC
		// does by including the cause.
		return fmt.Errorf("report %s %q: %v", what, err.Error(), cause)
	}

	logger := klog.FromContext(ctx)
	_, replyErr := p.restClient.Reply(ctx, &restproxyapi.ReplyMessage{Id: request.Id, Error: what + ": " + err.Error()})
	if replyErr != nil {
		logger.V(5).Info("Failed to send reply with error", "what", what, "err", err, "clientErr", replyErr)
		return fmt.Errorf("report %s %q: %v", what, err.Error(), replyErr)
	}
	logger.V(5).Info("Sent reply with error", "err", err)
	return nil
}

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
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"

	resourceapi "k8s.io/api/resource/v1alpha2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	restproxyapi "k8s.io/dynamic-resource-allocation/apis/restproxy/v1alpha1"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"

	_ "k8s.io/klog/v2/ktesting/init"
)

type testcase struct {
	handler func(ctx context.Context, tb testing.TB, w http.ResponseWriter, r *http.Request)
	grpc.UnaryServerInterceptor
	grpc.StreamServerInterceptor
	grpc.UnaryClientInterceptor
	grpc.StreamClientInterceptor
	filter RequestFilter

	do func(ctx context.Context, tb testing.TB, client http.Client)
}

type expectErrors struct {
	doErrorContains           string
	readResponseErrorContains string
}

func doOneRequest(expectErrors expectErrors, expectBody string, expectResponse func(t testing.TB, response http.Response)) func(ctx context.Context, t testing.TB, client http.Client) {
	return func(ctx context.Context, t testing.TB, client http.Client) {
		t.Helper()

		var req *http.Request
		var err error
		if ctx != nil {
			req, err = http.NewRequestWithContext(ctx, http.MethodGet, "http://no.such.server.example.com/my-path", strings.NewReader("ping"))
		} else {
			req, err = http.NewRequest(http.MethodGet, "http://no.such.server.example.com/my-path", strings.NewReader("ping"))
		}
		require.NoError(t, err, "new request")
		response, err := client.Do(req)
		if expectErrors.doErrorContains != "" {
			require.Error(t, err, "HTTP client")
			require.Contains(t, err.Error(), expectErrors.doErrorContains, "HTTP client")
			return
		} else {
			require.NoError(t, err, "HTTP client")
		}
		content, err := io.ReadAll(response.Body)
		if expectErrors.readResponseErrorContains != "" {
			require.Error(t, err, "response body")
			require.Contains(t, err.Error(), expectErrors.readResponseErrorContains, "response body")
			return
		} else {
			require.NoError(t, err, "response body")
		}
		require.Equal(t, expectBody, string(content), "response body")
		if expectResponse != nil {
			expectResponse(t, *response)
		}
	}
}

func TestProxy(t *testing.T) {
	var buffer strings.Builder
	buffer.Grow(60 * readChunkSize)
	pattern := fmt.Sprintf("line #%%d: %s\n", strings.Repeat("a", 70))
	i := 0

	for ; buffer.Len() < readChunkSize; i++ {
		buffer.WriteString(fmt.Sprintf(pattern, i))
	}
	singleMessage := buffer.String()

	// From kube-apiserver max message size (https://github.com/kubernetes/apiserver/blame/8d18eec7c050338aac4d49e470f3ea0b946f4726/pkg/server/config.go#L442).
	largeMessage := strings.Repeat("x", 3*1024*1024)

	for ; buffer.Len() < 5*readChunkSize; i++ {
		buffer.WriteString(fmt.Sprintf(pattern, i))
	}
	largeResponse := buffer.String()

	for i := 5 * readChunkSize / 10; buffer.Len() < 50*readChunkSize; i++ {
		buffer.WriteString(fmt.Sprintf(pattern, i))
	}
	largestResponse := buffer.String()

	testcases := map[string]testcase{
		"simple": {
			handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte("pong"))
			},
			do: doOneRequest(expectErrors{}, "pong", func(t testing.TB, actualResponse http.Response) {
				actualResponse.Request = nil
				actualResponse.Body = nil
				actualResponse.Header.Del("Date")
				expectedResponse := http.Response{
					Status:     "200 OK",
					StatusCode: 200,
					Proto:      "HTTP/1.1",
					ProtoMajor: 1,
					ProtoMinor: 1,
					Header: http.Header{
						"Content-Length": []string{"4"},
						"Content-Type":   []string{"text/plain; charset=utf-8"},
					},
				}
				assert.Equal(t, expectedResponse, actualResponse, "response")
			}),
		},
		"error": {
			handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusBadRequest)
			},
			do: doOneRequest(expectErrors{}, "", func(t testing.TB, response http.Response) {
				assert.Equal(t, "400 Bad Request", response.Status, "Status")
				assert.Equal(t, 400, response.StatusCode, "StatusCode")
			}),
		},
		"large": {
			handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(largeResponse))
			},
			do: doOneRequest(expectErrors{}, largeResponse, nil),
		},
		"largest": {
			handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(largestResponse))
			},
			do: doOneRequest(expectErrors{}, largestResponse, nil),
		},
		"random": {
			handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(http.StatusOK)
				offset := 0
				for offset < len(largeResponse) {
					size := rand.Intn(len(largeResponse) - offset)
					select {
					case <-ctx.Done():
					case <-time.After(time.Duration(rand.Intn(50)) * time.Millisecond):
					}
					if size == 0 {
						size = 1
					}
					_, _ = w.Write([]byte(largeResponse[offset : offset+size]))
					offset += size
				}
			},
			UnaryServerInterceptor: func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
				// Randomly delay delivering the gRPC call. Perhaps that'll cause
				// out-of-order Reply calls...
				select {
				case <-ctx.Done():
				case <-time.After(time.Duration(rand.Intn(50)) * time.Millisecond):
				}
				return handler(ctx, req)
			},
			do: doOneRequest(expectErrors{}, largeResponse, nil),
		},
		"many-streams": {
			handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(err.Error()))
					return
				}
				w.WriteHeader(http.StatusOK)
				for i := 0; i < 10; i++ {
					_, _ = w.Write(body)
				}
			},
			do: func(ctx context.Context, t testing.TB, client http.Client) {
				var wg sync.WaitGroup
				for x := 'a'; x < 'x'; x++ {
					x := x
					wg.Add(1)
					go func() {
						defer wg.Done()

						reqBody := strings.Repeat(string([]rune{x}), readChunkSize)
						req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://no.such.server.example.com/my-path", strings.NewReader(reqBody))
						if !assert.NoError(t, err, "new request") {
							return
						}

						response, err := client.Do(req)
						if !assert.NoError(t, err, "HTTP Get") {
							return
						}

						content, err := io.ReadAll(response.Body)
						if !assert.NoError(t, err, "read response") {
							return
						}
						assert.Equal(t, strings.Repeat(reqBody, 10), string(content), "response body")
					}()
				}
				wg.Wait()
			},
		},
		"meta-data": {
			handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(err.Error()))
					return
				}
				assert.Equal(t, http.MethodPut, r.Method, "method")
				assert.Equal(t, singleMessage, string(body), "request body")
				assert.Equal(t, "/my/complex/path", r.URL.Path, "path")
				assert.Subset(t, r.Header, http.Header{"Foo": []string{"1", "2"}, "X": []string{"y"}}, "header")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(body)
			},
			do: func(ctx context.Context, t testing.TB, client http.Client) {
				req, err := http.NewRequestWithContext(ctx, http.MethodPut, "http://no.such.server.example.com/my/complex/path", strings.NewReader(singleMessage))
				if !assert.NoError(t, err, "new request") {
					return
				}
				req.Header.Add("foo", "1")
				req.Header.Add("foo", "2")
				req.Header.Add("x", "y")

				response, err := client.Do(req)
				if !assert.NoError(t, err, "HTTP Put") {
					return
				}

				content, err := io.ReadAll(response.Body)
				if !assert.NoError(t, err, "read response") {
					return
				}
				assert.Equal(t, singleMessage, string(content), "response body")
			},
		},
		"large-request": {
			handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
				body, err := io.ReadAll(r.Body)
				if err != nil {
					w.WriteHeader(http.StatusInternalServerError)
					_, _ = w.Write([]byte(err.Error()))
					return
				}
				assert.Equal(t, largeMessage, string(body), "request body")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write(body)
			},
			do: func(ctx context.Context, t testing.TB, client http.Client) {
				req, err := http.NewRequestWithContext(ctx, http.MethodPut, "http://no.such.server.example.com/my/complex/path", strings.NewReader(largeMessage))
				if !assert.NoError(t, err, "new request") {
					return
				}
				response, err := client.Do(req)
				if !assert.NoError(t, err, "HTTP Put") {
					return
				}
				content, err := io.ReadAll(response.Body)
				if !assert.NoError(t, err, "read response") {
					return
				}
				assert.Equal(t, largeMessage, string(content), "response body")
			},
		},
	}

	for name, test := range testcases {
		test := test
		t.Run(name, func(t *testing.T) {
			t.Parallel()

			env := test.setup(t)
			test.do(env.ctx, t, env.client)
			t.Log("Test completed")
		})
	}
}

func TestRestartClient(t *testing.T) {
	test := testcase{
		handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("pong"))
		},
	}
	env := test.setup(t)
	ctx := env.ctx
	logger := klog.FromContext(ctx)

	doOneRequest(expectErrors{}, "pong", nil)(ctx, t, env.client)

	t.Log("Stopping REST client")
	env.roundTripper.Stop()
	env.grpcServer.GracefulStop()

	t.Log("Restarting REST client")
	roundTripperLogger := klog.LoggerWithName(logger, "RoundTripper2")
	roundTripperCtx := klog.NewContext(ctx, roundTripperLogger)
	roundTripper := StartRoundTripper(roundTripperCtx)
	env.unaryInterceptors[0] = UnaryContextInterceptor(roundTripperCtx)
	env.streamInterceptors[0] = StreamContextInterceptor(roundTripperCtx)
	t.Cleanup(roundTripper.Stop)
	grpcServer := grpc.NewServer(
		grpc.ChainUnaryInterceptor(env.unaryInterceptors...),
		grpc.ChainStreamInterceptor(env.streamInterceptors...),
	)
	restproxyapi.RegisterRESTServer(grpcServer, roundTripper)
	listener, err := net.ListenUnix("unix", &net.UnixAddr{Name: env.grpcSocket})
	require.NoError(t, err, "Unix domain socket listen")
	go func() {
		assert.NoError(t, grpcServer.Serve(listener), "gRPC server Serve")
	}()
	t.Cleanup(func() {
		roundTripper.Stop()
		grpcServer.GracefulStop()
	})
	client := http.Client{
		Transport: roundTripper,
	}

	doOneRequest(expectErrors{}, "pong", nil)(env.ctx, t, client)
	t.Log("Test completed")
}

func TestRestartProxy(t *testing.T) {
	test := testcase{
		handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("pong"))
		},
	}
	env := test.setup(t)
	ctx := env.ctx
	logger := klog.FromContext(ctx)

	doOneRequest(expectErrors{}, "pong", nil)(ctx, t, env.client)

	t.Log("Stopping REST proxy")
	env.restProxy.Stop()
	assert.NoError(t, env.grpcConn.Close(), "proxy client close")

	t.Log("Restarting REST proxy")
	restProxyLogger := klog.LoggerWithName(logger, "RESTProxy 2")
	restProxyCtx := klog.NewContext(ctx, restProxyLogger)
	grpcConn, err := grpc.DialContext(restProxyCtx, "unix://"+env.grpcSocket,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithChainUnaryInterceptor(env.unaryClientInterceptors...),
		grpc.WithChainStreamInterceptor(env.streamClientInterceptors...),
	)
	require.NoError(t, err, "gRPC dial")
	t.Cleanup(func() {
		assert.NoError(t, grpcConn.Close(), "proxy client close")
	})
	restProxy := StartRESTProxy(restProxyCtx, env.httpServerURL, env.httpServer.Client(), grpcConn, test.filter)
	t.Cleanup(restProxy.Stop)

	var response *http.Response
	require.EventuallyWithT(t, func(t *assert.CollectT) {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://no.such.server.example.com/my-path", strings.NewReader("ping"))
		require.NoError(t, err, "new request")
		response, err = env.client.Do(req)
		require.NoError(t, err, "client GET")
	}, 30*time.Second, time.Second, "successful GET")
	content, err := io.ReadAll(response.Body)
	require.NoError(t, err, "read body")
	require.Equal(t, "pong", string(content), "response body")
	t.Log("Test completed")
}

func TestNoHTTPServer(t *testing.T) {
	test := testcase{
		handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("pong"))
		},
	}
	env := test.setup(t)
	ctx := env.ctx

	env.httpServer.Close()
	doOneRequest(expectErrors{doErrorContains: "connection refused"}, "pong", nil)(ctx, t, env.client)
	t.Log("Test completed")
}

func TestIncompleteBody(t *testing.T) {
	cancelCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	readyCtx, ready := context.WithCancel(context.Background())
	defer ready()
	test := testcase{
		handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(strings.Repeat("a", readChunkSize)))
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}

			// Block to give the test time to close the client connection.
			ready()
			<-cancelCtx.Done()
		},
	}
	env := test.setup(t)
	ctx := env.ctx

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-readyCtx.Done()
		env.httpServer.CloseClientConnections()
		cancel()
	}()
	doOneRequest(expectErrors{readResponseErrorContains: "unexpected EOF"}, "pong", nil)(ctx, t, env.client)
	t.Log("Test completed")
}

func TestRequestCanceledDuringBody(t *testing.T) {
	cancelCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	readyCtx, ready := context.WithCancel(context.Background())
	defer ready()
	test := testcase{
		handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
			contentLength := 100 * readChunkSize
			w.Header().Add("Content-Length", fmt.Sprintf("%d", contentLength))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(strings.Repeat("a", contentLength/2)))
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}

			// Block to give the test time to cancel the request.
			ready()
			<-cancelCtx.Done()

			_, _ = w.Write([]byte(strings.Repeat("a", contentLength/2)))
		},
	}
	env := test.setup(t)
	ctx := env.ctx

	cancelReqCtx, cancelReq := context.WithCancelCause(ctx)
	defer cancelReq(errors.New("test completed"))

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-readyCtx.Done()
		t.Log("Canceling request")
		cancelReq(errors.New("testing request cancellation"))
	}()
	doOneRequest(expectErrors{readResponseErrorContains: "testing request cancellation"}, "pong", nil)(cancelReqCtx, t, env.client)
	wg.Wait()
	cancel()
	t.Log("Test completed")
}

func TestRequestCanceledBeforeHeader(t *testing.T) {
	cancelCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	readyCtx, ready := context.WithCancel(context.Background())
	defer ready()
	test := testcase{
		handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
			// Block to give the test time to cancel the request.
			ready()
			<-cancelCtx.Done()
		},
	}
	env := test.setup(t)
	ctx := env.ctx

	cancelReqCtx, cancelReq := context.WithCancelCause(ctx)
	defer cancelReq(errors.New("test completed"))

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-readyCtx.Done()
		t.Log("Canceling request")
		cancelReq(errors.New("testing request cancellation"))
	}()
	doOneRequest(expectErrors{doErrorContains: "testing request cancellation"}, "pong", nil)(cancelReqCtx, t, env.client)
	wg.Wait()
	cancel()
	t.Log("Test completed")
}

func TestProxyCanceledDuringBody(t *testing.T) {
	cancelCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	readyCtx, ready := context.WithCancel(context.Background())
	defer ready()
	test := testcase{
		handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
			contentLength := 100 * readChunkSize
			w.Header().Add("Content-Length", fmt.Sprintf("%d", contentLength))
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte(strings.Repeat("a", contentLength/2)))
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}

			// Block to give the test time to cancel the request.
			ready()
			<-cancelCtx.Done()

			_, _ = w.Write([]byte(strings.Repeat("a", contentLength/2)))
		},
	}
	env := test.setup(t)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-readyCtx.Done()
		t.Log("Canceling everything")
		env.cancel(errors.New("testing proxy cancellation"))

	}()
	doOneRequest(expectErrors{readResponseErrorContains: "testing proxy cancellation"}, "pong", nil)(nil, t, env.client)
	wg.Wait()
	cancel()
	t.Log("Test completed")
}

func TestProxyCanceledBeforeHeader(t *testing.T) {
	cancelCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	readyCtx, ready := context.WithCancel(context.Background())
	defer ready()
	test := testcase{
		handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
			// Block to give the test time to cancel the request.
			ready()
			<-cancelCtx.Done()
		},
	}
	env := test.setup(t)

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		<-readyCtx.Done()
		t.Log("Canceling everything")
		env.cancel(errors.New("testing proxy cancellation"))
	}()
	doOneRequest(expectErrors{doErrorContains: "testing proxy cancellation"}, "pong", nil)(nil, t, env.client)
	wg.Wait()
	cancel()
	t.Log("Test completed")
}

func TestRequestBodyClosed(t *testing.T) {
	cancelCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	readyCtx, ready := context.WithCancel(context.Background())
	defer ready()
	test := testcase{
		handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte("pong"))
			if flusher, ok := w.(http.Flusher); ok {
				flusher.Flush()
			}
			ready()
			<-cancelCtx.Done()
			_, _ = w.Write([]byte("boing"))
		},
	}
	env := test.setup(t)
	ctx := env.ctx

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, "http://no.such.server.example.com/my-path", strings.NewReader("ping"))
	require.NoError(t, err, "new request")
	response, err := env.client.Do(req)
	require.NoError(t, err, "HTTP client")

	<-readyCtx.Done()
	t.Log("Closing response body")
	require.NoError(t, response.Body.Close(), "close response body")
	cancel()

	// We cannot be sure when the REST proxy has processed the second Write.
	// Give it a second.
	time.Sleep(time.Second)

	t.Log("Test completed")
}

func TestResourceSliceList(t *testing.T) {
	expectedResourceSlices := &resourceapi.ResourceSliceList{
		Items: []resourceapi.ResourceSlice{{ObjectMeta: metav1.ObjectMeta{Name: "test"}}},
	}
	response, err := json.Marshal(expectedResourceSlices)
	require.NoError(t, err, "JSON encode of resource slice list")

	run := func(t *testing.T, options metav1.ListOptions, filter RequestFilter, expectQuery string) {
		test := testcase{
			handler: func(ctx context.Context, t testing.TB, w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/apis/resource.k8s.io/v1alpha2/resourceslices", r.URL.Path)
				assert.Equal(t, expectQuery, r.URL.RawQuery)
				w.Header().Add("Content-Encoding", "application/json")
				_, _ = w.Write(response)
			},
			filter: filter,
		}
		env := test.setup(t)
		ctx := env.ctx

		actualResourceSlices, err := env.clientset.ResourceV1alpha2().ResourceSlices().List(ctx, options)
		require.NoError(t, err)
		require.Equal(t, expectedResourceSlices, actualResourceSlices)
		require.NoError(t, err)
		require.Equal(t, expectedResourceSlices, actualResourceSlices)

		t.Log("Test completed")
	}

	filter := FilterDRADriver{
		NodeName:   "worker",
		DriverName: "test.example.org",
	}
	encodedQuery := "fieldSelector=driverName%3Dtest.example.org%2CnodeName%3Dworker"

	t.Run("without-options-no-filter", func(t *testing.T) {
		run(t, metav1.ListOptions{}, nil, "")
	})
	t.Run("with-options-no-filter", func(t *testing.T) {
		run(t, metav1.ListOptions{FieldSelector: "driverName=test.example.org,nodeName=worker"}, nil, encodedQuery)
	})
	t.Run("without-options-filter", func(t *testing.T) {
		run(t, metav1.ListOptions{}, filter, encodedQuery)
	})
	t.Run("with-options-filter", func(t *testing.T) {
		run(t, metav1.ListOptions{FieldSelector: "driverName=test.example.org,nodeName=worker"}, filter, encodedQuery)
	})
	t.Run("other-options", func(t *testing.T) {
		run(t, metav1.ListOptions{FieldSelector: "driverName=test2.example.org,nodeName=worker2,xyz=abc"}, filter, encodedQuery+"%2Cxyz%3Dabc")
	})
}

type testenv struct {
	ctx                      context.Context
	cancel                   func(error)
	grpcConn                 *grpc.ClientConn
	httpServer               *httptest.Server
	httpServerURL            *url.URL
	restProxy                *RESTProxy
	roundTripper             *RoundTripper
	grpcSocket               string
	unaryInterceptors        []grpc.UnaryServerInterceptor
	streamInterceptors       []grpc.StreamServerInterceptor
	unaryClientInterceptors  []grpc.UnaryClientInterceptor
	streamClientInterceptors []grpc.StreamClientInterceptor
	grpcServer               *grpc.Server
	client                   http.Client
	restConfig               *rest.Config
	clientset                kubernetes.Interface
}

func (test testcase) setup(t testing.TB) testenv {
	logger, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancelCause(ctx)
	t.Cleanup(func() { cancel(errors.New("test completed")) })

	httpServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		test.handler(ctx, t, w, r)
	}))
	t.Cleanup(func() { httpServer.Close() })

	tmp := t.TempDir()
	grpcSocket := path.Join(tmp, "gprc.sock")
	roundTripperLogger := klog.LoggerWithName(logger, "RoundTripper")
	roundTripperCtx := klog.NewContext(ctx, roundTripperLogger)
	roundTripper := StartRoundTripper(roundTripperCtx)
	t.Cleanup(func() { roundTripper.Stop() })
	listener, err := net.ListenUnix("unix", &net.UnixAddr{Name: grpcSocket})
	require.NoError(t, err, "Unix domain socket listen")
	unaryInterceptors := []grpc.UnaryServerInterceptor{
		UnaryContextInterceptor(roundTripperCtx),
		func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
			logger := klog.FromContext(ctx)
			logger.V(5).Info("gRPC server call", "method", info.FullMethod /*, "req", klog.Format(req) */)
			reply, err := handler(ctx, req)
			logger.V(5).Info("gRPC server response", "method", info.FullMethod /*, "reply", klog.Format(reply)*/, "err", err)
			return reply, err
		},
	}
	if test.UnaryServerInterceptor != nil {
		unaryInterceptors = append(unaryInterceptors, test.UnaryServerInterceptor)
	}
	streamInterceptors := []grpc.StreamServerInterceptor{
		StreamContextInterceptor(roundTripperCtx),
		func(srv any, ss grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
			ctx := ss.Context()
			logger := klog.FromContext(ctx)
			logger.V(5).Info("gRPC server stream start", "method", info.FullMethod)
			err := handler(srv, serverStreamLogger{logger: logger, method: info.FullMethod, ServerStream: ss})
			logger.V(5).Info("gRPC server stream end", "method", info.FullMethod, "err", err)
			return err
		},
	}
	if test.StreamServerInterceptor != nil {
		streamInterceptors = append(streamInterceptors, test.StreamServerInterceptor)
	}
	grpcServer := grpc.NewServer(
		grpc.ChainUnaryInterceptor(unaryInterceptors...),
		grpc.ChainStreamInterceptor(streamInterceptors...),
	)
	restproxyapi.RegisterRESTServer(grpcServer, roundTripper)
	go func() {
		assert.NoError(t, grpcServer.Serve(listener), "gRPC server Serve")
	}()
	t.Cleanup(func() {
		// We first need to close any stream, otherwise GracefulStop hangs.
		roundTripper.Stop()
		grpcServer.GracefulStop()
	})

	unaryClientInterceptors := []grpc.UnaryClientInterceptor{
		func(ctx context.Context, method string, req, reply any, cc *grpc.ClientConn, invoker grpc.UnaryInvoker, opts ...grpc.CallOption) error {
			logger := klog.FromContext(ctx)
			logger.V(5).Info("gRPC client call", "method", method /*, "req", klog.Format(req) */)
			err := invoker(ctx, method, req, reply, cc, opts...)
			logger.V(5).Info("gRPC client response", "method", method /*, "reply", klog.Format(reply) */, "err", err)
			return err
		},
	}
	if test.UnaryClientInterceptor != nil {
		unaryClientInterceptors = append(unaryClientInterceptors, test.UnaryClientInterceptor)
	}
	streamClientInterceptors := []grpc.StreamClientInterceptor{
		func(ctx context.Context, desc *grpc.StreamDesc, cc *grpc.ClientConn, method string, streamer grpc.Streamer, opts ...grpc.CallOption) (grpc.ClientStream, error) {
			logger := klog.FromContext(ctx)
			logger.V(5).Info("gRPC client stream start", "method", method, "streamName", desc.StreamName)
			stream, err := streamer(ctx, desc, cc, method, opts...)
			logger.V(5).Info("gRPC client stream end", "method", method, "streamName", desc.StreamName, "err", err)
			return clientStreamLogger{method: method, ClientStream: stream}, err
		},
	}
	if test.StreamClientInterceptor != nil {
		streamClientInterceptors = append(streamClientInterceptors, test.StreamClientInterceptor)
	}

	restProxyLogger := klog.LoggerWithName(logger, "RESTProxy")
	restProxyCtx := klog.NewContext(ctx, restProxyLogger)
	grpcConn, err := grpc.DialContext(restProxyCtx, "unix://"+grpcSocket,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithChainUnaryInterceptor(unaryClientInterceptors...),
		grpc.WithChainStreamInterceptor(streamClientInterceptors...),
	)
	require.NoError(t, err, "gRPC dial")
	t.Cleanup(func() {
		// Not quite idempotent... the second call returns an error.
		err := grpcConn.Close()
		if status.Code(err) == codes.Canceled {
			return
		}
		assert.NoError(t, err, "proxy client close")
	})
	httpServerURL, err := url.Parse(httpServer.URL)
	require.NoError(t, err, "HTTP server URL parsing")
	restProxy := StartRESTProxy(restProxyCtx, httpServerURL, httpServer.Client(), grpcConn, test.filter)
	t.Cleanup(restProxy.Stop)

	client := http.Client{
		Transport: roundTripper,
	}
	restConfig := roundTripper.NewRESTConfig()
	clientset, err := kubernetes.NewForConfig(restConfig)
	require.NoError(t, err, "create client set")

	return testenv{
		ctx:                      ctx,
		cancel:                   cancel,
		grpcConn:                 grpcConn,
		httpServerURL:            httpServerURL,
		httpServer:               httpServer,
		restProxy:                restProxy,
		roundTripper:             roundTripper,
		grpcSocket:               grpcSocket,
		unaryInterceptors:        unaryInterceptors,
		streamInterceptors:       streamInterceptors,
		unaryClientInterceptors:  unaryClientInterceptors,
		streamClientInterceptors: streamClientInterceptors,
		grpcServer:               grpcServer,
		client:                   client,
		restConfig:               restConfig,
		clientset:                clientset,
	}
}

type clientStreamLogger struct {
	method string
	grpc.ClientStream
}

func (c clientStreamLogger) SendMsg(m any) error {
	logger := klog.FromContext(c.Context())
	logger.V(5).Info("client stream send", "method", c.method, "message", klog.Format(m))
	err := c.ClientStream.SendMsg(m)
	logger.V(5).Info("client stream send done", "method", c.method, "err", err)
	return err
}

func (c clientStreamLogger) RecvMsg(m any) error {
	logger := klog.FromContext(c.Context())
	logger.V(5).Info("client stream recv", "method", c.method)
	err := c.ClientStream.RecvMsg(m)
	logger.V(5).Info("client stream recv done", "method", c.method /*, "message", klog.Format(m) */, "err", err)
	return err
}

type serverStreamLogger struct {
	logger klog.Logger
	method string
	grpc.ServerStream
}

func (c serverStreamLogger) SendMsg(m any) error {
	logger := c.logger
	logger.V(5).Info("server stream send", "method", c.method /*, "message", klog.Format(m) */)
	err := c.ServerStream.SendMsg(m)
	logger.V(5).Info("server stream send done", "method", c.method, "err", err)
	return err
}

func (c serverStreamLogger) RecvMsg(m any) error {
	logger := c.logger
	logger.V(5).Info("server stream recv", "method", c.method)
	err := c.ServerStream.RecvMsg(m)
	logger.V(5).Info("server stream recv done", "method", c.method /*, "message", klog.Format(m) */, "err", err)
	return err
}

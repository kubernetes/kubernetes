//go:build !windows

/*
Copyright 2017 The Kubernetes Authors.

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

// Package envelope transforms values for storage at rest using a Envelope provider
package envelope

import (
	"context"
	"fmt"
	"reflect"
	"sync"
	"testing"
	"time"

	mock "k8s.io/apiserver/pkg/storage/value/encrypt/envelope/testing/v1beta1"

	"k8s.io/apimachinery/pkg/util/uuid"
)

type testSocket struct {
	path     string
	endpoint string
}

// newEndpoint constructs a unique name for a Linux Abstract Socket to be used in a test.
// This package uses Linux Domain Sockets to remove the need for clean-up of socket files.
func newEndpoint() *testSocket {
	p := fmt.Sprintf("@%s.sock", uuid.NewUUID())

	return &testSocket{
		path:     p,
		endpoint: fmt.Sprintf("unix:///%s", p),
	}
}

// TestKMSPluginLateStart tests the scenario where kms-plugin pod/container starts after kube-apiserver pod/container.
// Since the Dial to kms-plugin is non-blocking we expect the construction of gRPC service to succeed even when
// kms-plugin is not yet up - dialing happens in the background.
func TestKMSPluginLateStart(t *testing.T) {
	t.Parallel()
	callTimeout := 3 * time.Second
	s := newEndpoint()

	ctx := testContext(t)

	service, err := NewGRPCService(ctx, s.endpoint, callTimeout)
	if err != nil {
		t.Fatalf("failed to create envelope service, error: %v", err)
	}
	defer destroyService(service)

	time.Sleep(callTimeout / 2)
	_ = mock.NewBase64Plugin(t, s.path)

	data := []byte("test data")
	_, err = service.Encrypt(data)
	if err != nil {
		t.Fatalf("failed when execute encrypt, error: %v", err)
	}
}

// TestTimeout tests behaviour of the kube-apiserver based on the supplied timeout and delayed start of kms-plugin.
func TestTimeouts(t *testing.T) {
	t.Parallel()
	var testCases = []struct {
		desc               string
		callTimeout        time.Duration
		pluginDelay        time.Duration
		kubeAPIServerDelay time.Duration
		wantErr            string
	}{
		{
			desc:        "timeout zero - expect failure when call from kube-apiserver arrives before plugin starts",
			callTimeout: 0 * time.Second,
			pluginDelay: 3 * time.Second,
			wantErr:     "rpc error: code = DeadlineExceeded desc = context deadline exceeded",
		},
		{
			desc:               "timeout zero but kms-plugin already up - still failure - zero timeout is an invalid value",
			callTimeout:        0 * time.Second,
			pluginDelay:        0 * time.Second,
			kubeAPIServerDelay: 2 * time.Second,
			wantErr:            "rpc error: code = DeadlineExceeded desc = context deadline exceeded",
		},
		{
			desc:        "timeout greater than kms-plugin delay - expect success",
			callTimeout: 6 * time.Second,
			pluginDelay: 3 * time.Second,
		},
		{
			desc:        "timeout less than kms-plugin delay - expect failure",
			callTimeout: 3 * time.Second,
			pluginDelay: 6 * time.Second,
			wantErr:     "rpc error: code = DeadlineExceeded desc = context deadline exceeded",
		},
	}

	for _, tt := range testCases {
		tt := tt
		t.Run(tt.desc, func(t *testing.T) {
			t.Parallel()
			var (
				service         Service
				err             error
				data            = []byte("test data")
				kubeAPIServerWG sync.WaitGroup
				kmsPluginWG     sync.WaitGroup
				testCompletedWG sync.WaitGroup
				socketName      = newEndpoint()
			)

			testCompletedWG.Add(1)
			defer testCompletedWG.Done()

			ctx := testContext(t)

			kubeAPIServerWG.Add(1)
			go func() {
				// Simulating late start of kube-apiserver - plugin is up before kube-apiserver, if requested by the testcase.
				time.Sleep(tt.kubeAPIServerDelay)

				service, err = NewGRPCService(ctx, socketName.endpoint, tt.callTimeout)
				if err != nil {
					t.Errorf("failed to create envelope service, error: %v", err)
					return
				}
				defer destroyService(service)
				kubeAPIServerWG.Done()
				// Keeping kube-apiserver up to process requests.
				testCompletedWG.Wait()
			}()

			kmsPluginWG.Add(1)
			go func() {
				// Simulating delayed start of kms-plugin, kube-apiserver is up before the plugin, if requested by the testcase.
				time.Sleep(tt.pluginDelay)

				_ = mock.NewBase64Plugin(t, socketName.path)

				kmsPluginWG.Done()
				// Keeping plugin up to process requests.
				testCompletedWG.Wait()
			}()

			kubeAPIServerWG.Wait()
			if t.Failed() {
				return
			}
			_, err = service.Encrypt(data)

			if err == nil && tt.wantErr != "" {
				t.Fatalf("got nil, want %s", tt.wantErr)
			}

			if err != nil && tt.wantErr == "" {
				t.Fatalf("got %q, want nil", err.Error())
			}

			// Collecting kms-plugin - allowing plugin to clean-up.
			kmsPluginWG.Wait()
		})
	}
}

// TestIntermittentConnectionLoss tests the scenario where the connection with kms-plugin is intermittently lost.
func TestIntermittentConnectionLoss(t *testing.T) {
	t.Parallel()
	var (
		wg1        sync.WaitGroup
		wg2        sync.WaitGroup
		timeout    = 30 * time.Second
		blackOut   = 1 * time.Second
		data       = []byte("test data")
		endpoint   = newEndpoint()
		encryptErr error
	)
	// Start KMS Plugin
	f := mock.NewBase64Plugin(t, endpoint.path)

	ctx := testContext(t)

	//  connect to kms plugin
	service, err := NewGRPCService(ctx, endpoint.endpoint, timeout)
	if err != nil {
		t.Fatalf("failed to create envelope service, error: %v", err)
	}
	defer destroyService(service)

	_, err = service.Encrypt(data)
	if err != nil {
		t.Fatalf("failed when execute encrypt, error: %v", err)
	}
	t.Log("Connected to KMSPlugin")
	f.CleanUp()

	// Stop KMS Plugin - simulating connection loss
	t.Log("KMS Plugin is stopping")
	time.Sleep(2 * time.Second)

	wg1.Add(1)
	wg2.Add(1)
	go func() {
		defer wg2.Done()
		// Call service to encrypt data.
		t.Log("Sending encrypt request")
		wg1.Done()
		_, err := service.Encrypt(data)
		if err != nil {
			encryptErr = fmt.Errorf("failed when executing encrypt, error: %v", err)
		}
	}()

	wg1.Wait()
	time.Sleep(blackOut)
	// Start KMS Plugin
	_ = mock.NewBase64Plugin(t, endpoint.path)
	t.Log("Restarted KMS Plugin")

	wg2.Wait()

	if encryptErr != nil {
		t.Error(encryptErr)
	}
}

func TestUnsupportedVersion(t *testing.T) {
	t.Parallel()
	ver := "invalid"
	data := []byte("test data")
	wantErr := fmt.Errorf(versionErrorf, ver, kmsapiVersion)
	endpoint := newEndpoint()

	f := mock.NewBase64Plugin(t, endpoint.path)
	f.SetVersion(ver)

	ctx := testContext(t)

	s, err := NewGRPCService(ctx, endpoint.endpoint, 1*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer destroyService(s)

	// Encrypt
	_, err = s.Encrypt(data)
	if err == nil || err.Error() != wantErr.Error() {
		t.Errorf("got err: %ver, want: %ver", err, wantErr)
	}

	destroyService(s)

	s, err = NewGRPCService(ctx, endpoint.endpoint, 1*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	defer destroyService(s)

	// Decrypt
	_, err = s.Decrypt(data)
	if err == nil || err.Error() != wantErr.Error() {
		t.Errorf("got err: %ver, want: %ver", err, wantErr)
	}
}

// Normal encryption and decryption operation.
func TestGRPCService(t *testing.T) {
	t.Parallel()
	// Start a test gRPC server.
	endpoint := newEndpoint()
	_ = mock.NewBase64Plugin(t, endpoint.path)

	ctx := testContext(t)

	// Create the gRPC client service.
	service, err := NewGRPCService(ctx, endpoint.endpoint, 1*time.Second)
	if err != nil {
		t.Fatalf("failed to create envelope service, error: %v", err)
	}
	defer destroyService(service)

	// Call service to encrypt data.
	data := []byte("test data")
	cipher, err := service.Encrypt(data)
	if err != nil {
		t.Fatalf("failed when execute encrypt, error: %v", err)
	}

	// Call service to decrypt data.
	result, err := service.Decrypt(cipher)
	if err != nil {
		t.Fatalf("failed when execute decrypt, error: %v", err)
	}

	if !reflect.DeepEqual(data, result) {
		t.Errorf("expect: %v, but: %v", data, result)
	}
}

// Normal encryption and decryption operation by multiple go-routines.
func TestGRPCServiceConcurrentAccess(t *testing.T) {
	t.Parallel()
	// Start a test gRPC server.
	endpoint := newEndpoint()
	_ = mock.NewBase64Plugin(t, endpoint.path)

	ctx := testContext(t)

	// Create the gRPC client service.
	service, err := NewGRPCService(ctx, endpoint.endpoint, 15*time.Second)
	if err != nil {
		t.Fatalf("failed to create envelope service, error: %v", err)
	}
	defer destroyService(service)

	var wg sync.WaitGroup
	n := 100
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			defer wg.Done()
			// Call service to encrypt data.
			data := []byte("test data")
			cipher, err := service.Encrypt(data)
			if err != nil {
				t.Errorf("failed when execute encrypt, error: %v", err)
			}

			// Call service to decrypt data.
			result, err := service.Decrypt(cipher)
			if err != nil {
				t.Errorf("failed when execute decrypt, error: %v", err)
			}

			if !reflect.DeepEqual(data, result) {
				t.Errorf("expect: %v, but: %v", data, result)
			}
		}()
	}

	wg.Wait()
}

func destroyService(service Service) {
	if service != nil {
		s := service.(*gRPCService)
		s.connection.Close()
	}
}

// Test all those invalid configuration for KMS provider.
func TestInvalidConfiguration(t *testing.T) {
	t.Parallel()
	// Start a test gRPC server.
	_ = mock.NewBase64Plugin(t, newEndpoint().path)

	ctx := testContext(t)

	invalidConfigs := []struct {
		name     string
		endpoint string
	}{
		{"emptyConfiguration", ""},
		{"invalidScheme", "tcp://localhost:6060"},
	}

	for _, testCase := range invalidConfigs {
		t.Run(testCase.name, func(t *testing.T) {
			_, err := NewGRPCService(ctx, testCase.endpoint, 1*time.Second)
			if err == nil {
				t.Fatalf("should fail to create envelope service for %s.", testCase.name)
			}
		})
	}
}

func testContext(t *testing.T) context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	return ctx
}

/*
Copyright 2019 The Kubernetes Authors.

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

package egressselector

import (
	"context"
	"net"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apiserver/pkg/apis/apiserver"
)

type fakeEgressSelection struct {
	directDialerCalled bool
}

func TestEgressSelector(t *testing.T) {
	testcases := []struct {
		name     string
		input    *apiserver.EgressSelectorConfiguration
		services []struct {
			egressType     EgressType
			validateDialer func(dialer utilnet.DialFunc, s *fakeEgressSelection) (bool, error)
			lookupError    *string
			dialerError    *string
		}
		expectedError *string
	}{
		{
			name: "direct",
			input: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "cluster",
						Connection: apiserver.Connection{
							Type: "direct",
							HTTPConnect: &apiserver.HTTPConnectConfig{
								URL:        "",
								CABundle:   "",
								ClientKey:  "",
								ClientCert: "",
							},
						},
					},
					{
						Name: "master",
						Connection: apiserver.Connection{
							Type: "direct",
							HTTPConnect: &apiserver.HTTPConnectConfig{
								URL:        "",
								CABundle:   "",
								ClientKey:  "",
								ClientCert: "",
							},
						},
					},
					{
						Name: "etcd",
						Connection: apiserver.Connection{
							Type: "direct",
							HTTPConnect: &apiserver.HTTPConnectConfig{
								URL:        "",
								CABundle:   "",
								ClientKey:  "",
								ClientCert: "",
							},
						},
					},
				},
			},
			services: []struct {
				egressType     EgressType
				validateDialer func(dialer utilnet.DialFunc, s *fakeEgressSelection) (bool, error)
				lookupError    *string
				dialerError    *string
			}{
				{
					Cluster,
					validateDirectDialer,
					nil,
					nil,
				},
				{
					Master,
					validateDirectDialer,
					nil,
					nil,
				},
				{
					Etcd,
					validateDirectDialer,
					nil,
					nil,
				},
			},
			expectedError: nil,
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			// Setup the various pieces such as the fake dialer prior to initializing the egress selector.
			// Go doesn't allow function pointer comparison, nor does its reflect package
			// So overriding the default dialer to detect if it is returned.
			fake := &fakeEgressSelection{}
			directDialer = fake.fakeDirectDialer
			cs, err := NewEgressSelector(tc.input)
			if err == nil && tc.expectedError != nil {
				t.Errorf("calling NewEgressSelector expected error: %s, did not get it", *tc.expectedError)
			}
			if err != nil && tc.expectedError == nil {
				t.Errorf("unexpected error calling NewEgressSelector got: %#v", err)
			}
			if err != nil && tc.expectedError != nil && err.Error() != *tc.expectedError {
				t.Errorf("calling NewEgressSelector expected error: %s, got %#v", *tc.expectedError, err)
			}

			for _, service := range tc.services {
				networkContext := NetworkContext{EgressSelectionName: service.egressType}
				dialer, lookupErr := cs.Lookup(networkContext)
				if lookupErr == nil && service.lookupError != nil {
					t.Errorf("calling Lookup expected error: %s, did not get it", *service.lookupError)
				}
				if lookupErr != nil && service.lookupError == nil {
					t.Errorf("unexpected error calling Lookup got: %#v", lookupErr)
				}
				if lookupErr != nil && service.lookupError != nil && lookupErr.Error() != *service.lookupError {
					t.Errorf("calling Lookup expected error: %s, got %#v", *service.lookupError, lookupErr)
				}
				fake.directDialerCalled = false
				ok, dialerErr := service.validateDialer(dialer, fake)
				if dialerErr == nil && service.dialerError != nil {
					t.Errorf("calling Lookup expected error: %s, did not get it", *service.dialerError)
				}
				if dialerErr != nil && service.dialerError == nil {
					t.Errorf("unexpected error calling Lookup got: %#v", dialerErr)
				}
				if dialerErr != nil && service.dialerError != nil && dialerErr.Error() != *service.dialerError {
					t.Errorf("calling Lookup expected error: %s, got %#v", *service.dialerError, dialerErr)
				}
				if !ok {
					t.Errorf("Could not validate dialer for service %q", service.egressType)
				}
			}
		})
	}
}

func (s *fakeEgressSelection) fakeDirectDialer(ctx context.Context, network, address string) (net.Conn, error) {
	s.directDialerCalled = true
	return nil, nil
}

func validateDirectDialer(dialer utilnet.DialFunc, s *fakeEgressSelection) (bool, error) {
	conn, err := dialer(context.Background(), "tcp", "127.0.0.1:8080")
	if err != nil {
		return false, err
	}
	if conn != nil {
		return false, nil
	}
	return s.directDialerCalled, nil
}

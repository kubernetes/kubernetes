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

package options

import (
	"fmt"
	"net"
	"strconv"
	"testing"

	"k8s.io/apimachinery/pkg/util/rand"
	apiserveroptions "k8s.io/apiserver/pkg/server/options"
	schedulerappconfig "k8s.io/kubernetes/cmd/kube-scheduler/app/config"
	kubeschedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
)

func TestOptions_ApplyTo(t *testing.T) {
	tests := []struct {
		name                                                              string
		options                                                           Options
		configLoaded                                                      bool
		expectHealthzBindAddress, expectMetricsBindAddress                string
		expectInsecureServingAddress, expectInsecureMetricsServingAddress string
		expectInsecureServingPort, expectInsecureMetricsServingPort       int
		wantErr                                                           bool
	}{
		{
			name: "no config, zero port",
			options: Options{
				ComponentConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
					HealthzBindAddress: "1.2.3.4:1234",
					MetricsBindAddress: "1.2.3.4:1234",
				},
				CombinedInsecureServing: &CombinedInsecureServingOptions{
					Healthz:  (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					Metrics:  (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					BindPort: 0,
				},
			},
			configLoaded: false,
		},
		{
			name: "config loaded, non-nil healthz",
			options: Options{
				ComponentConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
					HealthzBindAddress: "1.2.3.4:1234",
					MetricsBindAddress: "1.2.3.4:1234",
				},
				CombinedInsecureServing: &CombinedInsecureServingOptions{
					Healthz:  (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					BindPort: 0,
				},
			},
			configLoaded: true,

			expectHealthzBindAddress:     "1.2.3.4:1234",
			expectInsecureServingPort:    1234,
			expectInsecureServingAddress: "1.2.3.4",
		},
		{
			name: "config loaded, non-nil metrics",
			options: Options{
				ComponentConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
					HealthzBindAddress: "1.2.3.4:1234",
					MetricsBindAddress: "1.2.3.4:1234",
				},
				CombinedInsecureServing: &CombinedInsecureServingOptions{
					Metrics:  (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					BindPort: 0,
				},
			},
			configLoaded: true,

			expectMetricsBindAddress:            "1.2.3.4:1234",
			expectInsecureMetricsServingPort:    1234,
			expectInsecureMetricsServingAddress: "1.2.3.4",
		},
		{
			name: "config loaded, all set, zero BindPort",
			options: Options{
				ComponentConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
					HealthzBindAddress: "1.2.3.4:1234",
					MetricsBindAddress: "1.2.3.4:1234",
				},
				CombinedInsecureServing: &CombinedInsecureServingOptions{
					Healthz:  (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					Metrics:  (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					BindPort: 0,
				},
			},
			configLoaded: true,

			expectHealthzBindAddress:     "1.2.3.4:1234",
			expectInsecureServingPort:    1234,
			expectInsecureServingAddress: "1.2.3.4",

			expectMetricsBindAddress: "1.2.3.4:1234",
		},
		{
			name: "config loaded, all set, different addresses",
			options: Options{
				ComponentConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
					HealthzBindAddress: "1.2.3.4:1234",
					MetricsBindAddress: "1.2.3.4:1235",
				},
				CombinedInsecureServing: &CombinedInsecureServingOptions{
					Healthz:  (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					Metrics:  (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					BindPort: 0,
				},
			},
			configLoaded: true,

			expectHealthzBindAddress:     "1.2.3.4:1234",
			expectInsecureServingPort:    1234,
			expectInsecureServingAddress: "1.2.3.4",

			expectMetricsBindAddress:            "1.2.3.4:1235",
			expectInsecureMetricsServingPort:    1235,
			expectInsecureMetricsServingAddress: "1.2.3.4",
		},
		{
			name: "no config, all set, port passed",
			options: Options{
				ComponentConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
					HealthzBindAddress: "1.2.3.4:1234",
					MetricsBindAddress: "1.2.3.4:1234",
				},
				CombinedInsecureServing: &CombinedInsecureServingOptions{
					Healthz:     (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					Metrics:     (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					BindPort:    1236,
					BindAddress: "1.2.3.4",
				},
			},
			configLoaded: false,

			expectHealthzBindAddress:     "1.2.3.4:1236",
			expectInsecureServingPort:    1236,
			expectInsecureServingAddress: "1.2.3.4",

			expectMetricsBindAddress: "1.2.3.4:1236",
		},
		{
			name: "no config, all set, address passed",
			options: Options{
				ComponentConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
					HealthzBindAddress: "1.2.3.4:1234",
					MetricsBindAddress: "1.2.3.4:1234",
				},
				CombinedInsecureServing: &CombinedInsecureServingOptions{
					Healthz:     (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					Metrics:     (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					BindAddress: "2.3.4.5",
					BindPort:    1234,
				},
			},
			configLoaded: false,

			expectHealthzBindAddress:     "2.3.4.5:1234",
			expectInsecureServingPort:    1234,
			expectInsecureServingAddress: "2.3.4.5",

			expectMetricsBindAddress: "2.3.4.5:1234",
		},
		{
			name: "no config, all set, zero port passed",
			options: Options{
				ComponentConfig: kubeschedulerconfig.KubeSchedulerConfiguration{
					HealthzBindAddress: "1.2.3.4:1234",
					MetricsBindAddress: "1.2.3.4:1234",
				},
				CombinedInsecureServing: &CombinedInsecureServingOptions{
					Healthz:     (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					Metrics:     (&apiserveroptions.DeprecatedInsecureServingOptions{}).WithLoopback(),
					BindAddress: "2.3.4.5",
					BindPort:    0,
				},
			},
			configLoaded: false,
		},
	}
	for i, tt := range tests {
		t.Run(fmt.Sprintf("%d-%s", i, tt.name), func(t *testing.T) {
			c := schedulerappconfig.Config{
				ComponentConfig: tt.options.ComponentConfig,
			}

			if tt.options.CombinedInsecureServing != nil {
				if tt.options.CombinedInsecureServing.Healthz != nil {
					tt.options.CombinedInsecureServing.Healthz.ListenFunc = createMockListener
				}
				if tt.options.CombinedInsecureServing.Metrics != nil {
					tt.options.CombinedInsecureServing.Metrics.ListenFunc = createMockListener
				}
			}

			if tt.configLoaded {
				if err := tt.options.CombinedInsecureServing.ApplyToFromLoadedConfig(&c, &c.ComponentConfig); (err != nil) != tt.wantErr {
					t.Fatalf("%d - Options.ApplyTo() error = %v, wantErr %v", i, err, tt.wantErr)
				}
			} else {
				if err := tt.options.CombinedInsecureServing.ApplyTo(&c, &c.ComponentConfig); (err != nil) != tt.wantErr {
					t.Fatalf("%d - Options.ApplyTo() error = %v, wantErr %v", i, err, tt.wantErr)
				}
			}
			if got, expect := c.ComponentConfig.HealthzBindAddress, tt.expectHealthzBindAddress; got != expect {
				t.Errorf("%d - expected HealthzBindAddress %q, got %q", i, expect, got)
			}
			if got, expect := c.ComponentConfig.MetricsBindAddress, tt.expectMetricsBindAddress; got != expect {
				t.Errorf("%d - expected MetricsBindAddress %q, got %q", i, expect, got)
			}
			if got, expect := c.InsecureServing != nil, tt.expectInsecureServingPort != 0; got != expect {
				t.Errorf("%d - expected InsecureServing != nil to be %v, got %v", i, expect, got)
			} else if c.InsecureServing != nil {
				if got, expect := c.InsecureServing.Listener.(*mockListener).address, tt.expectInsecureServingAddress; got != expect {
					t.Errorf("%d - expected healthz address %q, got %q", i, expect, got)
				}
				if got, expect := c.InsecureServing.Listener.(*mockListener).port, tt.expectInsecureServingPort; got != expect {
					t.Errorf("%d - expected healthz port %v, got %v", i, expect, got)
				}
			}
			if got, expect := c.InsecureMetricsServing != nil, tt.expectInsecureMetricsServingPort != 0; got != expect {
				t.Errorf("%d - expected Metrics != nil to be %v, got %v", i, expect, got)
			} else if c.InsecureMetricsServing != nil {
				if got, expect := c.InsecureMetricsServing.Listener.(*mockListener).address, tt.expectInsecureMetricsServingAddress; got != expect {
					t.Errorf("%d - expected metrics address %q, got %q", i, expect, got)
				}
				if got, expect := c.InsecureMetricsServing.Listener.(*mockListener).port, tt.expectInsecureMetricsServingPort; got != expect {
					t.Errorf("%d - expected metrics port %v, got %v", i, expect, got)
				}
			}
		})
	}
}

type mockListener struct {
	address string
	port    int
}

func createMockListener(network, addr string, config net.ListenConfig) (net.Listener, int, error) {
	host, portInt, err := splitHostIntPort(addr)
	if err != nil {
		return nil, 0, err
	}
	if portInt == 0 {
		portInt = rand.IntnRange(1, 32767)
	}
	return &mockListener{host, portInt}, portInt, nil
}

func (l *mockListener) Accept() (net.Conn, error) { return nil, nil }
func (l *mockListener) Close() error              { return nil }
func (l *mockListener) Addr() net.Addr {
	return mockAddr(net.JoinHostPort(l.address, strconv.Itoa(l.port)))
}

type mockAddr string

func (a mockAddr) Network() string { return "tcp" }
func (a mockAddr) String() string  { return string(a) }

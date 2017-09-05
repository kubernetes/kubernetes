/*
Copyright 2014 The Kubernetes Authors.

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

package openstack

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

func TestReadConfig(t *testing.T) {
	_, err := readConfig(nil)
	if err == nil {
		t.Errorf("Should fail when no config is provided: %s", err)
	}

	cfg, err := readConfig(strings.NewReader(`
 [Global]
 auth-url = http://auth.url
 username = user
 [LoadBalancer]
 create-monitor = yes
 monitor-delay = 1m
 monitor-timeout = 30s
 monitor-max-retries = 3
 [BlockStorage]
 bs-version = auto
 trust-device-path = yes

 `))
	if err != nil {
		t.Fatalf("Should succeed when a valid config is provided: %s", err)
	}
	if cfg.Global.AuthUrl != "http://auth.url" {
		t.Errorf("incorrect authurl: %s", cfg.Global.AuthUrl)
	}

	if !cfg.LoadBalancer.CreateMonitor {
		t.Errorf("incorrect lb.createmonitor: %t", cfg.LoadBalancer.CreateMonitor)
	}
	if cfg.LoadBalancer.MonitorDelay.Duration != 1*time.Minute {
		t.Errorf("incorrect lb.monitordelay: %s", cfg.LoadBalancer.MonitorDelay)
	}
	if cfg.LoadBalancer.MonitorTimeout.Duration != 30*time.Second {
		t.Errorf("incorrect lb.monitortimeout: %s", cfg.LoadBalancer.MonitorTimeout)
	}
	if cfg.LoadBalancer.MonitorMaxRetries != 3 {
		t.Errorf("incorrect lb.monitormaxretries: %d", cfg.LoadBalancer.MonitorMaxRetries)
	}
	if cfg.BlockStorage.TrustDevicePath != true {
		t.Errorf("incorrect bs.trustdevicepath: %v", cfg.BlockStorage.TrustDevicePath)
	}
	if cfg.BlockStorage.BSVersion != "auto" {
		t.Errorf("incorrect bs.bs-version: %v", cfg.BlockStorage.BSVersion)
	}
}

func TestToAuthOptions(t *testing.T) {
	cfg := Config{}
	cfg.Global.Username = "user"
	// etc.

	ao := cfg.toAuthOptions()

	if !ao.AllowReauth {
		t.Errorf("Will need to be able to reauthenticate")
	}
	if ao.Username != cfg.Global.Username {
		t.Errorf("Username %s != %s", ao.Username, cfg.Global.Username)
	}
}

func TestCheckOpenStackOpts(t *testing.T) {
	delay := MyDuration{60 * time.Second}
	timeout := MyDuration{30 * time.Second}
	tests := []struct {
		name          string
		openstackOpts *OpenStack
		expectedError error
	}{
		{
			name: "test1",
			openstackOpts: &OpenStack{
				provider: nil,
				lbOpts: LoadBalancerOpts{
					LBVersion:            "v2",
					SubnetId:             "6261548e-ffde-4bc7-bd22-59c83578c5ef",
					FloatingNetworkId:    "38b8b5f9-64dc-4424-bf86-679595714786",
					LBMethod:             "ROUND_ROBIN",
					CreateMonitor:        true,
					MonitorDelay:         delay,
					MonitorTimeout:       timeout,
					MonitorMaxRetries:    uint(3),
					ManageSecurityGroups: true,
					NodeSecurityGroupID:  "b41d28c2-d02f-4e1e-8ffb-23b8e4f5c144",
				},
			},
			expectedError: nil,
		},
		{
			name: "test2",
			openstackOpts: &OpenStack{
				provider: nil,
				lbOpts: LoadBalancerOpts{
					LBVersion:            "v2",
					FloatingNetworkId:    "38b8b5f9-64dc-4424-bf86-679595714786",
					LBMethod:             "ROUND_ROBIN",
					CreateMonitor:        true,
					MonitorDelay:         delay,
					MonitorTimeout:       timeout,
					MonitorMaxRetries:    uint(3),
					ManageSecurityGroups: true,
					NodeSecurityGroupID:  "b41d28c2-d02f-4e1e-8ffb-23b8e4f5c144",
				},
			},
			expectedError: nil,
		},
		{
			name: "test3",
			openstackOpts: &OpenStack{
				provider: nil,
				lbOpts: LoadBalancerOpts{
					LBVersion:            "v2",
					SubnetId:             "6261548e-ffde-4bc7-bd22-59c83578c5ef",
					FloatingNetworkId:    "38b8b5f9-64dc-4424-bf86-679595714786",
					LBMethod:             "ROUND_ROBIN",
					CreateMonitor:        true,
					ManageSecurityGroups: true,
					NodeSecurityGroupID:  "b41d28c2-d02f-4e1e-8ffb-23b8e4f5c144",
				},
			},
			expectedError: fmt.Errorf("monitor-delay not set in cloud provider config"),
		},
		{
			name: "test4",
			openstackOpts: &OpenStack{
				provider: nil,
				lbOpts: LoadBalancerOpts{
					LBVersion:            "v2",
					SubnetId:             "6261548e-ffde-4bc7-bd22-59c83578c5ef",
					FloatingNetworkId:    "38b8b5f9-64dc-4424-bf86-679595714786",
					LBMethod:             "ROUND_ROBIN",
					CreateMonitor:        true,
					MonitorDelay:         delay,
					MonitorTimeout:       timeout,
					MonitorMaxRetries:    uint(3),
					ManageSecurityGroups: true,
				},
			},
			expectedError: fmt.Errorf("node-security-group not set in cloud provider config"),
		},
	}

	for _, testcase := range tests {
		err := checkOpenStackOpts(testcase.openstackOpts)

		if err == nil && testcase.expectedError == nil {
			continue
		}
		if (err != nil && testcase.expectedError == nil) || (err == nil && testcase.expectedError != nil) || err.Error() != testcase.expectedError.Error() {
			t.Errorf("%s failed: expected err=%q, got %q",
				testcase.name, testcase.expectedError, err)
		}
	}
}

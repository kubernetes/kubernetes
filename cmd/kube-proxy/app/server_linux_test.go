//go:build linux

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

package app

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/spf13/pflag"
	kubeproxyconfig "k8s.io/kubernetes/pkg/proxy/apis/config"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func Test_platformApplyDefaults(t *testing.T) {
	testCases := []struct {
		name                string
		mode                kubeproxyconfig.ProxyMode
		expectedMode        kubeproxyconfig.ProxyMode
		detectLocal         kubeproxyconfig.LocalMode
		expectedDetectLocal kubeproxyconfig.LocalMode
	}{
		{
			name:                "defaults",
			mode:                "",
			expectedMode:        kubeproxyconfig.ProxyModeIPTables,
			detectLocal:         "",
			expectedDetectLocal: kubeproxyconfig.LocalModeClusterCIDR,
		},
		{
			name:                "explicit",
			mode:                kubeproxyconfig.ProxyModeIPTables,
			expectedMode:        kubeproxyconfig.ProxyModeIPTables,
			detectLocal:         kubeproxyconfig.LocalModeClusterCIDR,
			expectedDetectLocal: kubeproxyconfig.LocalModeClusterCIDR,
		},
		{
			name:                "override mode",
			mode:                "ipvs",
			expectedMode:        kubeproxyconfig.ProxyModeIPVS,
			detectLocal:         "",
			expectedDetectLocal: kubeproxyconfig.LocalModeClusterCIDR,
		},
		{
			name:                "override detect-local",
			mode:                "",
			expectedMode:        kubeproxyconfig.ProxyModeIPTables,
			detectLocal:         "NodeCIDR",
			expectedDetectLocal: kubeproxyconfig.LocalModeNodeCIDR,
		},
		{
			name:                "override both",
			mode:                "ipvs",
			expectedMode:        kubeproxyconfig.ProxyModeIPVS,
			detectLocal:         "NodeCIDR",
			expectedDetectLocal: kubeproxyconfig.LocalModeNodeCIDR,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			options := NewOptions()
			config := &kubeproxyconfig.KubeProxyConfiguration{
				Mode:            tc.mode,
				DetectLocalMode: tc.detectLocal,
			}

			options.platformApplyDefaults(config)
			if config.Mode != tc.expectedMode {
				t.Fatalf("expected mode: %s, but got: %s", tc.expectedMode, config.Mode)
			}
			if config.DetectLocalMode != tc.expectedDetectLocal {
				t.Fatalf("expected detect-local: %s, but got: %s", tc.expectedDetectLocal, config.DetectLocalMode)
			}
		})
	}
}

func TestConfigChange(t *testing.T) {
	setUp := func() (*os.File, string, error) {
		tempDir, err := os.MkdirTemp("", "kubeproxy-config-change")
		if err != nil {
			return nil, "", fmt.Errorf("unable to create temporary directory: %v", err)
		}
		fullPath := filepath.Join(tempDir, "kube-proxy-config")
		file, err := os.Create(fullPath)
		if err != nil {
			return nil, "", fmt.Errorf("unexpected error when creating temp file: %v", err)
		}

		_, err = file.WriteString(`apiVersion: kubeproxy.config.k8s.io/v1alpha1
bindAddress: 0.0.0.0
bindAddressHardFail: false
clientConnection:
  acceptContentTypes: ""
  burst: 10
  contentType: application/vnd.kubernetes.protobuf
  kubeconfig: /var/lib/kube-proxy/kubeconfig.conf
  qps: 5
clusterCIDR: 10.244.0.0/16
configSyncPeriod: 15m0s
conntrack:
  maxPerCore: 32768
  min: 131072
  tcpCloseWaitTimeout: 1h0m0s
  tcpEstablishedTimeout: 24h0m0s
enableProfiling: false
healthzBindAddress: 0.0.0.0:10256
hostnameOverride: ""
iptables:
  masqueradeAll: false
  masqueradeBit: 14
  minSyncPeriod: 0s
  syncPeriod: 30s
ipvs:
  excludeCIDRs: null
  minSyncPeriod: 0s
  scheduler: ""
  syncPeriod: 30s
kind: KubeProxyConfiguration
metricsBindAddress: 127.0.0.1:10249
mode: ""
nodePortAddresses: null
oomScoreAdj: -999
portRange: ""
detectLocalMode: "BridgeInterface"`)
		if err != nil {
			return nil, "", fmt.Errorf("unexpected error when writing content to temp kube-proxy config file: %v", err)
		}

		return file, tempDir, nil
	}

	tearDown := func(file *os.File, tempDir string) {
		file.Close()
		os.RemoveAll(tempDir)
	}

	testCases := []struct {
		name        string
		proxyServer proxyRun
		append      bool
		expectedErr string
	}{
		{
			name:        "update config file",
			proxyServer: new(fakeProxyServerLongRun),
			append:      true,
			expectedErr: "content of the proxy server's configuration file was updated",
		},
		{
			name:        "fake error",
			proxyServer: new(fakeProxyServerError),
			expectedErr: "mocking error from ProxyServer.Run()",
		},
	}

	for _, tc := range testCases {
		_, ctx := ktesting.NewTestContext(t)
		file, tempDir, err := setUp()
		if err != nil {
			t.Fatalf("unexpected error when setting up environment: %v", err)
		}

		opt := NewOptions()
		opt.ConfigFile = file.Name()
		err = opt.Complete(new(pflag.FlagSet))
		if err != nil {
			t.Fatal(err)
		}
		opt.proxyServer = tc.proxyServer

		errCh := make(chan error, 1)
		go func() {
			errCh <- opt.runLoop(ctx)
		}()

		if tc.append {
			file.WriteString("append fake content")
		}

		select {
		case err := <-errCh:
			if err != nil {
				if !strings.Contains(err.Error(), tc.expectedErr) {
					t.Errorf("[%s] Expected error containing %v, got %v", tc.name, tc.expectedErr, err)
				}
			}
		case <-time.After(10 * time.Second):
			t.Errorf("[%s] Timeout: unable to get any events or internal timeout.", tc.name)
		}
		tearDown(file, tempDir)
	}
}

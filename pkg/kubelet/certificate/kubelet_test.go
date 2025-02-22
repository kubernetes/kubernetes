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

package certificate

import (
	"bytes"
	"context"
	"crypto/x509"
	"crypto/x509/pkix"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/util/cert"
	cloudproviderapi "k8s.io/cloud-provider/api"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	netutils "k8s.io/utils/net"
)

func TestAddressesToHostnamesAndIPs(t *testing.T) {
	tests := []struct {
		name         string
		addresses    []v1.NodeAddress
		wantDNSNames []string
		wantIPs      []net.IP
	}{
		{
			name:         "empty",
			addresses:    nil,
			wantDNSNames: nil,
			wantIPs:      nil,
		},
		{
			name:         "ignore empty values",
			addresses:    []v1.NodeAddress{{Type: v1.NodeHostName, Address: ""}},
			wantDNSNames: nil,
			wantIPs:      nil,
		},
		{
			name: "ignore invalid IPs",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.2"},
				{Type: v1.NodeExternalIP, Address: "3.4"},
			},
			wantDNSNames: nil,
			wantIPs:      nil,
		},
		{
			name: "dedupe values",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeHostName, Address: "hostname"},
				{Type: v1.NodeExternalDNS, Address: "hostname"},
				{Type: v1.NodeInternalDNS, Address: "hostname"},
				{Type: v1.NodeInternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
			},
			wantDNSNames: []string{"hostname"},
			wantIPs:      []net.IP{netutils.ParseIPSloppy("1.1.1.1")},
		},
		{
			name: "order values",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeHostName, Address: "hostname-2"},
				{Type: v1.NodeExternalDNS, Address: "hostname-1"},
				{Type: v1.NodeInternalDNS, Address: "hostname-3"},
				{Type: v1.NodeInternalIP, Address: "2.2.2.2"},
				{Type: v1.NodeExternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "3.3.3.3"},
			},
			wantDNSNames: []string{"hostname-1", "hostname-2", "hostname-3"},
			wantIPs:      []net.IP{netutils.ParseIPSloppy("1.1.1.1"), netutils.ParseIPSloppy("2.2.2.2"), netutils.ParseIPSloppy("3.3.3.3")},
		},
		{
			name: "handle IP and DNS hostnames",
			addresses: []v1.NodeAddress{
				{Type: v1.NodeHostName, Address: "hostname"},
				{Type: v1.NodeHostName, Address: "1.1.1.1"},
			},
			wantDNSNames: []string{"hostname"},
			wantIPs:      []net.IP{netutils.ParseIPSloppy("1.1.1.1")},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotDNSNames, gotIPs := addressesToHostnamesAndIPs(tt.addresses)
			if !reflect.DeepEqual(gotDNSNames, tt.wantDNSNames) {
				t.Errorf("addressesToHostnamesAndIPs() gotDNSNames = %v, want %v", gotDNSNames, tt.wantDNSNames)
			}
			if !reflect.DeepEqual(gotIPs, tt.wantIPs) {
				t.Errorf("addressesToHostnamesAndIPs() gotIPs = %v, want %v", gotIPs, tt.wantIPs)
			}
		})
	}
}

func removeThenCreate(name string, data []byte, perm os.FileMode) error {
	if err := os.Remove(name); err != nil {
		if !os.IsNotExist(err) {
			return err
		}
	}
	return os.WriteFile(name, data, perm)
}

func createCertAndKeyFiles(certDir string) (string, string, error) {
	cert, key, err := cert.GenerateSelfSignedCertKey("k8s.io", nil, nil)
	if err != nil {
		return "", "", nil
	}

	certPath := filepath.Join(certDir, "kubelet.cert")
	keyPath := filepath.Join(certDir, "kubelet.key")
	if err := removeThenCreate(certPath, cert, os.FileMode(0644)); err != nil {
		return "", "", err
	}

	if err := removeThenCreate(keyPath, key, os.FileMode(0600)); err != nil {
		return "", "", err
	}

	return certPath, keyPath, nil
}

// createCertAndKeyFilesUsingRename creates cert and key files under a parent dir `identity` as
// <certDir>/identity/kubelet.cert, <certDir>/identity/kubelet.key
func createCertAndKeyFilesUsingRename(certDir string) (string, string, error) {
	cert, key, err := cert.GenerateSelfSignedCertKey("k8s.io", nil, nil)
	if err != nil {
		return "", "", nil
	}

	var certKeyPathFn = func(dataDir string) (string, string, string) {
		outputDir := filepath.Join(certDir, dataDir)
		return outputDir, filepath.Join(outputDir, "kubelet.cert"), filepath.Join(outputDir, "kubelet.key")
	}

	writeDir, writeCertPath, writeKeyPath := certKeyPathFn("identity.tmp")
	if err := os.Mkdir(writeDir, 0777); err != nil {
		return "", "", err
	}

	if err := removeThenCreate(writeCertPath, cert, os.FileMode(0644)); err != nil {
		return "", "", err
	}

	if err := removeThenCreate(writeKeyPath, key, os.FileMode(0600)); err != nil {
		return "", "", err
	}

	targetDir, certPath, keyPath := certKeyPathFn("identity")
	if err := os.RemoveAll(targetDir); err != nil {
		if !os.IsNotExist(err) {
			return "", "", err
		}
	}
	if err := os.Rename(writeDir, targetDir); err != nil {
		return "", "", err
	}

	return certPath, keyPath, nil
}

func TestKubeletServerCertificateFromFiles(t *testing.T) {
	// test two common ways of certificate file updates:
	// 1. delete and write the cert and key files directly
	// 2. create the cert and key files under a child dir and perform dir rename during update
	tests := []struct {
		name      string
		useRename bool
	}{
		{
			name:      "remove and create",
			useRename: false,
		},
		{
			name:      "rename cert dir",
			useRename: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			createFn := createCertAndKeyFiles
			if tt.useRename {
				createFn = createCertAndKeyFilesUsingRename
			}

			certDir := t.TempDir()
			certPath, keyPath, err := createFn(certDir)
			if err != nil {
				t.Fatalf("Unable to setup cert files: %v", err)
			}

			m, err := NewKubeletServerCertificateDynamicFileManager(certPath, keyPath)
			if err != nil {
				t.Fatalf("Unable to create certificte provider: %v", err)
			}

			m.Start()
			defer m.Stop()

			c := m.Current()
			if c == nil {
				t.Fatal("failed to provide valid certificate")
			}
			time.Sleep(100 * time.Millisecond)
			c2 := m.Current()
			if c2 == nil {
				t.Fatal("failed to provide valid certificate")
			}
			if c2 != c {
				t.Errorf("expected the same loaded certificate object when there is no cert file change, got different")
			}

			// simulate certificate files updated in the background
			if _, _, err := createFn(certDir); err != nil {
				t.Fatalf("got errors when rotating certificate files in the test: %v", err)
			}

			err = wait.PollUntilContextTimeout(context.Background(),
				100*time.Millisecond, 10*time.Second, true,
				func(_ context.Context) (bool, error) {
					c3 := m.Current()
					if c3 == nil {
						return false, fmt.Errorf("expected valid certificate regardless of file changes, but got nil")
					}
					if bytes.Equal(c.Certificate[0], c3.Certificate[0]) {
						t.Logf("loaded certificate is not updated")
						return false, nil
					}
					return true, nil
				})
			if err != nil {
				t.Errorf("failed to provide the updated certificate after file changes: %v", err)
			}

			if err = os.Remove(certPath); err != nil {
				t.Errorf("could not delete file in order to perform test")
			}

			time.Sleep(1 * time.Second)
			if m.Current() == nil {
				t.Errorf("expected the manager still provides cached content when certificate file was not available")
			}
		})
	}
}

func TestNewCertificateManagerConfigGetTemplate(t *testing.T) {
	nodeName := "fake-node"
	nodeIP := netutils.ParseIPSloppy("192.168.1.1")
	tests := []struct {
		name          string
		nodeAddresses []v1.NodeAddress
		want          *x509.CertificateRequest
		featuregate   bool
	}{
		{
			name:        "node addresses or hostnames and gate enabled",
			featuregate: true,
		},
		{
			name:        "node addresses or hostnames and gate disabled",
			featuregate: false,
		},
		{
			name: "only hostnames and gate enabled",
			nodeAddresses: []v1.NodeAddress{
				{
					Type:    v1.NodeHostName,
					Address: nodeName,
				},
			},
			want: &x509.CertificateRequest{
				Subject: pkix.Name{
					CommonName:   fmt.Sprintf("system:node:%s", nodeName),
					Organization: []string{"system:nodes"},
				},
				DNSNames: []string{nodeName},
			},
			featuregate: true,
		},
		{
			name: "only hostnames and gate disabled",
			nodeAddresses: []v1.NodeAddress{
				{
					Type:    v1.NodeHostName,
					Address: nodeName,
				},
			},
			featuregate: false,
		},
		{
			name: "only IP addresses and gate enabled",
			nodeAddresses: []v1.NodeAddress{
				{
					Type:    v1.NodeInternalIP,
					Address: nodeIP.String(),
				},
			},
			want: &x509.CertificateRequest{
				Subject: pkix.Name{
					CommonName:   fmt.Sprintf("system:node:%s", nodeName),
					Organization: []string{"system:nodes"},
				},
				IPAddresses: []net.IP{nodeIP},
			},
			featuregate: true,
		},
		{
			name: "only IP addresses and gate disabled",
			nodeAddresses: []v1.NodeAddress{
				{
					Type:    v1.NodeInternalIP,
					Address: nodeIP.String(),
				},
			},
			want: &x509.CertificateRequest{
				Subject: pkix.Name{
					CommonName:   fmt.Sprintf("system:node:%s", nodeName),
					Organization: []string{"system:nodes"},
				},
				IPAddresses: []net.IP{nodeIP},
			},
			featuregate: false,
		},
		{
			name: "IP addresses and hostnames and gate enabled",
			nodeAddresses: []v1.NodeAddress{
				{
					Type:    v1.NodeHostName,
					Address: nodeName,
				},
				{
					Type:    v1.NodeInternalIP,
					Address: nodeIP.String(),
				},
			},
			want: &x509.CertificateRequest{
				Subject: pkix.Name{
					CommonName:   fmt.Sprintf("system:node:%s", nodeName),
					Organization: []string{"system:nodes"},
				},
				DNSNames:    []string{nodeName},
				IPAddresses: []net.IP{nodeIP},
			},
			featuregate: true,
		},
		{
			name: "IP addresses and hostnames and gate disabled",
			nodeAddresses: []v1.NodeAddress{
				{
					Type:    v1.NodeHostName,
					Address: nodeName,
				},
				{
					Type:    v1.NodeInternalIP,
					Address: nodeIP.String(),
				},
			},
			want: &x509.CertificateRequest{
				Subject: pkix.Name{
					CommonName:   fmt.Sprintf("system:node:%s", nodeName),
					Organization: []string{"system:nodes"},
				},
				DNSNames:    []string{nodeName},
				IPAddresses: []net.IP{nodeIP},
			},
			featuregate: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.AllowDNSOnlyNodeCSR, tt.featuregate)
			getAddresses := func() []v1.NodeAddress {
				return tt.nodeAddresses
			}
			getTemplate := newGetTemplateFn(types.NodeName(nodeName), getAddresses)
			got := getTemplate()
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Wrong certificate, got %v expected %v", got, tt.want)
				return
			}
		})
	}
}

func TestGetNodeAddressesFromInformer(t *testing.T) {
	testCases := []struct {
		name          string
		nodeName      types.NodeName
		node          *v1.Node
		expectedAddrs []v1.NodeAddress
	}{
		{
			name:          "node not found",
			nodeName:      "test-node",
			node:          nil,
			expectedAddrs: nil,
		},
		{
			name:     "empty addresses",
			nodeName: "test-node",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
				Status:     v1.NodeStatus{Addresses: []v1.NodeAddress{}},
			},
			expectedAddrs: nil,
		},
		{
			name:     "no taints",
			nodeName: "test-node",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
				Status: v1.NodeStatus{
					Addresses: []v1.NodeAddress{
						{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
						{Type: v1.NodeHostName, Address: "test-node"},
					},
				},
			},
			expectedAddrs: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
				{Type: v1.NodeHostName, Address: "test-node"},
			},
		},
		{
			name:     "external cloud provider taint",
			nodeName: "test-node",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{Key: cloudproviderapi.TaintExternalCloudProvider, Effect: v1.TaintEffectNoSchedule},
					},
				},
				Status: v1.NodeStatus{
					Addresses: []v1.NodeAddress{
						{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
						{Type: v1.NodeHostName, Address: "test-node"},
					},
				},
			},
			expectedAddrs: nil,
		},
		{
			name:     "other taint",
			nodeName: "test-node",
			node: &v1.Node{
				ObjectMeta: metav1.ObjectMeta{Name: "test-node"},
				Spec: v1.NodeSpec{
					Taints: []v1.Taint{
						{Key: "other-taint", Effect: v1.TaintEffectNoSchedule},
					},
				},
				Status: v1.NodeStatus{
					Addresses: []v1.NodeAddress{
						{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
						{Type: v1.NodeHostName, Address: "test-node"},
					},
				},
			},
			expectedAddrs: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
				{Type: v1.NodeHostName, Address: "test-node"},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			if tc.node != nil {
				_, err := client.CoreV1().Nodes().Create(context.TODO(), tc.node, metav1.CreateOptions{})
				if err != nil {
					t.Fatalf("failed to create node: %v", err)
				}
			}
			kubeInformers := informers.NewSharedInformerFactory(client, 0)
			nodeLister := kubeInformers.Core().V1().Nodes().Lister()
			kubeInformers.Start(nil)
			kubeInformers.WaitForCacheSync(nil)

			addrs := getNodeAddressesFromInformer(tc.nodeName, nodeLister)

			if len(addrs) != len(tc.expectedAddrs) {
				t.Errorf("expected %d addresses, got %d", len(tc.expectedAddrs), len(addrs))
			} else {
				for i := range addrs {
					if addrs[i] != tc.expectedAddrs[i] {
						t.Errorf("expected address %v, got %v", tc.expectedAddrs[i], addrs[i])
					}
				}
			}
		})
	}
}

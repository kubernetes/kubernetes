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

package cloudprovider

import (
	"context"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
	cloudprovider "k8s.io/cloud-provider"
	cloudproviderapi "k8s.io/cloud-provider/api"
	ccmservertesting "k8s.io/cloud-provider/app/testing"
	fakecloud "k8s.io/cloud-provider/fake"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	"k8s.io/kubernetes/test/integration/framework"
)

func Test_RemoveExternalCloudProviderTaint(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, []string{"--disable-admission-plugins=ServiceAccount"}, framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)

	ns := framework.CreateNamespaceOrDie(client, "config-map", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	// Create fake node
	_, err := client.CoreV1().Nodes().Create(ctx, makeNode("node0"), metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Failed to create Node %v", err)
	}

	// start cloud-controller-manager
	kubeconfig := createKubeconfigFileForRestConfig(server.ClientConfig)
	// nolint:errcheck // Ignore the error trying to delete the kubeconfig file used for the test
	defer os.Remove(kubeconfig)
	args := []string{
		"--kubeconfig=" + kubeconfig,
		"--cloud-provider=fakeCloudTaints",
		"--cidr-allocator-type=" + string(ipam.RangeAllocatorType),
		"--configure-cloud-routes=false",
	}

	fakeCloud := &fakecloud.Cloud{
		Zone: cloudprovider.Zone{
			FailureDomain: "zone-0",
			Region:        "region-1",
		},
		EnableInstancesV2:  true,
		ExistsByProviderID: true,
		ProviderID: map[types.NodeName]string{
			types.NodeName("node0"): "12345",
		},
		InstanceTypes: map[types.NodeName]string{
			types.NodeName("node0"): "t1.micro",
		},
		ExtID: map[types.NodeName]string{
			types.NodeName("node0"): "12345",
		},
		Addresses: []v1.NodeAddress{
			{
				Type:    v1.NodeHostName,
				Address: "node0.cloud.internal",
			},
			{
				Type:    v1.NodeInternalIP,
				Address: "10.0.0.1",
			},
			{
				Type:    v1.NodeInternalIP,
				Address: "192.168.0.1",
			},
			{
				Type:    v1.NodeExternalIP,
				Address: "132.143.154.163",
			},
		},
		ErrByProviderID: nil,
		Err:             nil,
	}

	// register fake GCE cloud provider
	cloudprovider.RegisterCloudProvider(
		"fakeCloudTaints",
		func(config io.Reader) (cloudprovider.Interface, error) {
			return fakeCloud, nil
		})

	ccm := ccmservertesting.StartTestServerOrDie(ctx, args)
	defer ccm.TearDownFn()

	// There should be only the taint TaintNodeNotReady, added by the admission plugin TaintNodesByCondition
	err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 50*time.Second, true, func(ctx context.Context) (done bool, err error) {
		n, err := client.CoreV1().Nodes().Get(ctx, "node0", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		if len(n.Spec.Taints) != 1 {
			return false, nil
		}
		if n.Spec.Taints[0].Key != v1.TaintNodeNotReady {
			return false, nil
		}
		if len(n.Status.Addresses) != 4 {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Logf("Fake Cloud Provider calls: %v", fakeCloud.Calls)
		t.Fatalf("expected node to not have Taint: %v", err)
	}
}

// Test the behavior of the alpha.kubernetes.io/provided-node-ip annotation
// and the external cloud provider.
func Test_ExternalCloudProviderNodeAddresses(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
	server := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())
	defer server.TearDownFn()

	client := clientset.NewForConfigOrDie(server.ClientConfig)

	ns := framework.CreateNamespaceOrDie(client, "config-map", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	// start cloud-controller-manager
	kubeconfig := createKubeconfigFileForRestConfig(server.ClientConfig)
	// nolint:errcheck // Ignore the error trying to delete the kubeconfig file used for the test
	defer os.Remove(kubeconfig)
	args := []string{
		"--kubeconfig=" + kubeconfig,
		"--cloud-provider=fakeCloud",
		"--cidr-allocator-type=" + string(ipam.RangeAllocatorType),
		"--configure-cloud-routes=false",
	}
	originalAddresses := []v1.NodeAddress{
		{
			Type:    v1.NodeHostName,
			Address: "node.cloud.internal",
		},
		{
			Type:    v1.NodeInternalIP,
			Address: "10.0.0.1",
		},
		{
			Type:    v1.NodeInternalIP,
			Address: "172.16.0.1",
		},
		{
			Type:    v1.NodeInternalIP,
			Address: "fd00:1:2:3:4::",
		},
		{
			Type:    v1.NodeInternalIP,
			Address: "192.168.0.1",
		},
		{
			Type:    v1.NodeInternalIP,
			Address: "2001:db2::1",
		},
		{
			Type:    v1.NodeExternalIP,
			Address: "132.143.154.163",
		},
	}

	fakeCloud := &fakecloud.Cloud{
		Zone: cloudprovider.Zone{
			FailureDomain: "zone-0",
			Region:        "region-1",
		},
		EnableInstancesV2:  true,
		ExistsByProviderID: true,
		ProviderID: map[types.NodeName]string{
			types.NodeName("node-0"): "12345",
			types.NodeName("node-1"): "12345",
			types.NodeName("node-2"): "12345",
			types.NodeName("node-3"): "12345",
			types.NodeName("node-4"): "12345",
		},
		Addresses:       originalAddresses,
		ErrByProviderID: nil,
		Err:             nil,
	}
	// register fake GCE cloud provider
	cloudprovider.RegisterCloudProvider(
		"fakeCloud",
		func(config io.Reader) (cloudprovider.Interface, error) {
			return fakeCloud, nil
		})
	ccm := ccmservertesting.StartTestServerOrDie(ctx, args)
	defer ccm.TearDownFn()

	testCases := []struct {
		name    string
		nodeIPs string
	}{
		{
			name:    "IPv4",
			nodeIPs: "192.168.0.1",
		},
		{
			name:    "IPv6",
			nodeIPs: "2001:db2::1",
		},
		{
			name:    "IPv6-IPv4",
			nodeIPs: "2001:db2::1,172.16.0.1",
		},
		{
			name:    "IPv4-IPv6",
			nodeIPs: "192.168.0.1,fd00:1:2:3:4::",
		},
	}

	for d, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			nodeName := fmt.Sprintf("node-%d", d)

			// Create fake node
			node := makeNode(nodeName)
			node.Annotations = map[string]string{cloudproviderapi.AnnotationAlphaProvidedIPAddr: tc.nodeIPs}
			_, err := client.CoreV1().Nodes().Create(ctx, node, metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("Failed to create Node %v", err)
			}
			defer func() {
				err := client.CoreV1().Nodes().Delete(ctx, node.Name, metav1.DeleteOptions{})
				if err != nil {
					t.Fatalf("Failed to delete Node %v", err)
				}
			}()
			// There should be only the taint TaintNodeNotReady, added by the admission plugin TaintNodesByCondition
			err = wait.PollUntilContextTimeout(ctx, 1*time.Second, 50*time.Second, true, func(ctx context.Context) (done bool, err error) {
				n, err := client.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
				if err != nil {
					return false, err
				}
				if len(n.Spec.Taints) != 1 {
					return false, nil
				}
				if n.Spec.Taints[0].Key != v1.TaintNodeNotReady {
					return false, nil
				}

				gotInternalIPs := []string{}
				for _, address := range n.Status.Addresses {
					if address.Type == v1.NodeInternalIP {
						gotInternalIPs = append(gotInternalIPs, address.Address)
					}
				}
				nodeIPs := strings.Split(tc.nodeIPs, ",")
				// validate only the passed IP as annotation is present
				if !reflect.DeepEqual(gotInternalIPs, nodeIPs) {
					t.Logf("got node InternalIPs: %v expected node InternalIPs: %v", gotInternalIPs, nodeIPs)
					return false, nil
				}

				return true, nil
			})
			if err != nil {
				t.Logf("Fake Cloud Provider calls: %v", fakeCloud.Calls)
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// sigs.k8s.io/controller-runtime/pkg/envtest
func createKubeconfigFileForRestConfig(restConfig *rest.Config) string {
	clusters := make(map[string]*clientcmdapi.Cluster)
	clusters["default-cluster"] = &clientcmdapi.Cluster{
		Server:                   restConfig.Host,
		TLSServerName:            restConfig.ServerName,
		CertificateAuthorityData: restConfig.CAData,
	}
	contexts := make(map[string]*clientcmdapi.Context)
	contexts["default-context"] = &clientcmdapi.Context{
		Cluster:  "default-cluster",
		AuthInfo: "default-user",
	}
	authinfos := make(map[string]*clientcmdapi.AuthInfo)
	authinfos["default-user"] = &clientcmdapi.AuthInfo{
		ClientCertificateData: restConfig.CertData,
		ClientKeyData:         restConfig.KeyData,
		Token:                 restConfig.BearerToken,
	}
	clientConfig := clientcmdapi.Config{
		Kind:           "Config",
		APIVersion:     "v1",
		Clusters:       clusters,
		Contexts:       contexts,
		CurrentContext: "default-context",
		AuthInfos:      authinfos,
	}
	kubeConfigFile, _ := os.CreateTemp("", "kubeconfig")
	_ = clientcmd.WriteToFile(clientConfig, kubeConfigFile.Name())
	return kubeConfigFile.Name()
}

func makeNode(name string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.NodeSpec{
			Taints: []v1.Taint{{
				Key:    cloudproviderapi.TaintExternalCloudProvider,
				Value:  "true",
				Effect: v1.TaintEffectNoSchedule,
			}},
			Unschedulable: false,
		},
		Status: v1.NodeStatus{
			Conditions: []v1.NodeCondition{
				{
					Type:              v1.NodeReady,
					Status:            v1.ConditionUnknown,
					LastHeartbeatTime: metav1.Time{Time: time.Now()},
				},
			},
		},
	}
}

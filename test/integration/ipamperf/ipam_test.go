//go:build !providerless
// +build !providerless

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

package ipamperf

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os"
	"testing"
	"time"

	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	netutils "k8s.io/utils/net"

	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controller/nodeipam"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/integration/util"
)

func setupAllocator(ctx context.Context, kubeConfig *restclient.Config, config *Config, clusterCIDR, serviceCIDR *net.IPNet, subnetMaskSize int) (*clientset.Clientset, util.ShutdownFunc, error) {
	controllerStopChan := make(chan struct{})
	shutdownFunc := func() {
		close(controllerStopChan)
	}

	clientConfig := restclient.CopyConfig(kubeConfig)
	clientConfig.QPS = float32(config.KubeQPS)
	clientConfig.Burst = config.KubeQPS
	clientSet := clientset.NewForConfigOrDie(clientConfig)

	sharedInformer := informers.NewSharedInformerFactory(clientSet, 1*time.Hour)
	ipamController, err := nodeipam.NewNodeIpamController(
		ctx,
		sharedInformer.Core().V1().Nodes(),
		config.Cloud, clientSet, []*net.IPNet{clusterCIDR}, serviceCIDR, nil,
		[]int{subnetMaskSize}, config.AllocatorType,
	)
	if err != nil {
		return nil, shutdownFunc, err
	}
	go ipamController.Run(ctx)
	sharedInformer.Start(controllerStopChan)

	return clientSet, shutdownFunc, nil
}

func runTest(t *testing.T, kubeConfig *restclient.Config, config *Config, clusterCIDR, serviceCIDR *net.IPNet, subnetMaskSize int) (*Results, error) {
	t.Helper()
	klog.Infof("Running test %s", t.Name())

	nodeClientConfig := restclient.CopyConfig(kubeConfig)
	nodeClientConfig.QPS = float32(config.CreateQPS)
	nodeClientConfig.Burst = config.CreateQPS
	nodeClient := clientset.NewForConfigOrDie(nodeClientConfig)

	defer deleteNodes(nodeClient) // cleanup nodes on after controller shutdown
	_, ctx := ktesting.NewTestContext(t)
	clientSet, shutdownFunc, err := setupAllocator(ctx, kubeConfig, config, clusterCIDR, serviceCIDR, subnetMaskSize)
	if err != nil {
		t.Fatalf("Error starting IPAM allocator: %v", err)
	}
	defer shutdownFunc()

	o := NewObserver(clientSet, config.NumNodes)
	if err := o.StartObserving(); err != nil {
		t.Fatalf("Could not start test observer: %v", err)
	}

	if err := createNodes(nodeClient, config); err != nil {
		t.Fatalf("Could not create nodes: %v", err)
	}

	results := o.Results(t.Name(), config)
	klog.Infof("Results: %s", results)
	if !results.Succeeded {
		t.Errorf("%s: Not allocations succeeded", t.Name())
	}
	return results, nil
}

func logResults(allResults []*Results) {
	jStr, err := json.MarshalIndent(allResults, "", "  ")
	if err != nil {
		klog.Errorf("Error formatting results: %v", err)
		return
	}
	if resultsLogFile != "" {
		klog.Infof("Logging results to %s", resultsLogFile)
		if err := os.WriteFile(resultsLogFile, jStr, os.FileMode(0644)); err != nil {
			klog.Errorf("Error logging results to %s: %v", resultsLogFile, err)
		}
	}
	klog.Infof("AllResults:\n%s", string(jStr))
}

func TestPerformance(t *testing.T) {
	// TODO (#93112) skip test until appropriate timeout established
	if testing.Short() || true {
		// TODO (#61854) find why flakiness is caused by etcd connectivity before enabling always
		t.Skip("Skipping because we want to run short tests")
	}

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	_, kubeConfig, tearDownFn := framework.StartTestServer(ctx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount", "TaintNodesByCondition"}
		},
	})
	defer tearDownFn()

	_, clusterCIDR, _ := netutils.ParseCIDRSloppy("10.96.0.0/11") // allows up to 8K nodes
	_, serviceCIDR, _ := netutils.ParseCIDRSloppy("10.94.0.0/24") // does not matter for test - pick upto  250 services
	subnetMaskSize := 24

	var (
		allResults []*Results
		tests      []*Config
	)

	if isCustom {
		tests = append(tests, customConfig)
	} else {
		for _, numNodes := range []int{10, 100} {
			for _, alloc := range []ipam.CIDRAllocatorType{ipam.RangeAllocatorType, ipam.CloudAllocatorType, ipam.IPAMFromClusterAllocatorType, ipam.IPAMFromCloudAllocatorType} {
				tests = append(tests, &Config{AllocatorType: alloc, NumNodes: numNodes, CreateQPS: numNodes, KubeQPS: 10, CloudQPS: 10})
			}
		}
	}

	for _, test := range tests {
		testName := fmt.Sprintf("%s-KubeQPS%d-Nodes%d", test.AllocatorType, test.KubeQPS, test.NumNodes)
		t.Run(testName, func(t *testing.T) {
			allocateCIDR := false
			if test.AllocatorType == ipam.IPAMFromCloudAllocatorType || test.AllocatorType == ipam.CloudAllocatorType {
				allocateCIDR = true
			}
			bil := newBaseInstanceList(allocateCIDR, clusterCIDR, subnetMaskSize)
			cloud, err := util.NewMockGCECloud(bil.newMockCloud())
			if err != nil {
				t.Fatalf("Unable to create mock cloud: %v", err)
			}
			test.Cloud = cloud
			if results, err := runTest(t, kubeConfig, test, clusterCIDR, serviceCIDR, subnetMaskSize); err == nil {
				allResults = append(allResults, results)
			}
		})
	}

	logResults(allResults)
}

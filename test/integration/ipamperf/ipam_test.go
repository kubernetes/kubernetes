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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"testing"
	"time"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/controller/nodeipam"
	"k8s.io/kubernetes/pkg/controller/nodeipam/ipam"
	"k8s.io/kubernetes/test/integration/util"
)

func setupAllocator(apiURL string, config *Config, clusterCIDR, serviceCIDR *net.IPNet, subnetMaskSize int) (*clientset.Clientset, util.ShutdownFunc, error) {
	controllerStopChan := make(chan struct{})
	shutdownFunc := func() {
		close(controllerStopChan)
	}

	clientSet := clientset.NewForConfigOrDie(&restclient.Config{
		Host:          apiURL,
		ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Groups[v1.GroupName].GroupVersion()},
		QPS:           float32(config.KubeQPS),
		Burst:         config.KubeQPS,
	})

	sharedInformer := informers.NewSharedInformerFactory(clientSet, 1*time.Hour)
	ipamController, err := nodeipam.NewNodeIpamController(
		sharedInformer.Core().V1().Nodes(), config.Cloud, clientSet,
		clusterCIDR, serviceCIDR, subnetMaskSize, true, config.AllocatorType,
	)
	if err != nil {
		return nil, shutdownFunc, err
	}
	go ipamController.Run(controllerStopChan)
	sharedInformer.Start(controllerStopChan)

	return clientSet, shutdownFunc, nil
}

func runTest(t *testing.T, apiURL string, config *Config, clusterCIDR, serviceCIDR *net.IPNet, subnetMaskSize int) (*Results, error) {
	glog.Infof("Running test %s", t.Name())

	defer deleteNodes(apiURL, config) // cleanup nodes on after controller shutdown

	clientSet, shutdownFunc, err := setupAllocator(apiURL, config, clusterCIDR, serviceCIDR, subnetMaskSize)
	if err != nil {
		t.Fatalf("Error starting IPAM allocator: %v", err)
	}
	defer shutdownFunc()

	o := NewObserver(clientSet, config.NumNodes)
	err = o.StartObserving()
	if err != nil {
		t.Fatalf("Could not start test observer: %v", err)
	}

	err = createNodes(apiURL, config)
	if err != nil {
		t.Fatalf("Could not create nodes: %v", err)
	}

	results := o.Results(t.Name(), config)
	glog.Infof("Results: %s", results)
	if !results.Succeeded {
		t.Errorf("%s: Not allocations succeeded", t.Name())
	}
	return results, nil
}

func logResults(allResults []*Results) {
	jStr, err := json.MarshalIndent(allResults, "", "  ")
	if err != nil {
		glog.Errorf("Error formating results: %v", err)
		return
	}
	if resultsLogFile != "" {
		glog.Infof("Logging results to %s", resultsLogFile)
		if err := ioutil.WriteFile(resultsLogFile, jStr, os.FileMode(0644)); err != nil {
			glog.Errorf("Error logging results to %s: %v", resultsLogFile, err)
		}
	}
	glog.Infof("AllResults:\n%s", string(jStr))
}

func TestPerformance(t *testing.T) {
	apiURL, masterShutdown := util.StartApiserver()
	defer masterShutdown()

	_, clusterCIDR, _ := net.ParseCIDR("10.96.0.0/11")
	_, serviceCIDR, _ := net.ParseCIDR("10.94.0.0/18")
	subnetMaskSize := 24

	allResults := []*Results{}

	tests := []*Config{
		{AllocatorType: ipam.RangeAllocatorType, NumNodes: 10, CreateQPS: 10, KubeQPS: 10, CloudQPS: 10},
		{AllocatorType: ipam.IPAMFromClusterAllocatorType, NumNodes: 10, CreateQPS: 10, KubeQPS: 10, CloudQPS: 10},
		{AllocatorType: ipam.CloudAllocatorType, NumNodes: 10, CreateQPS: 10, KubeQPS: 10, CloudQPS: 10},
		{AllocatorType: ipam.IPAMFromCloudAllocatorType, NumNodes: 10, CreateQPS: 10, KubeQPS: 10, CloudQPS: 10},
		{AllocatorType: ipam.RangeAllocatorType, NumNodes: 100, CreateQPS: 100, KubeQPS: 10, CloudQPS: 10},
		{AllocatorType: ipam.IPAMFromClusterAllocatorType, NumNodes: 100, CreateQPS: 100, KubeQPS: 10, CloudQPS: 10},
		{AllocatorType: ipam.CloudAllocatorType, NumNodes: 100, CreateQPS: 100, KubeQPS: 10, CloudQPS: 10},
		{AllocatorType: ipam.IPAMFromCloudAllocatorType, NumNodes: 100, CreateQPS: 100, KubeQPS: 10, CloudQPS: 10},
	}

	for _, test := range tests {
		testName := fmt.Sprintf("%sKubeQPS%dNodes%d", test.AllocatorType, test.KubeQPS, test.NumNodes)
		t.Run(testName, func(t *testing.T) {
			var cs *cloudServer
			switch test.AllocatorType {
			case ipam.CloudAllocatorType:
				fallthrough
			case ipam.IPAMFromCloudAllocatorType:
				cs = newCloudServerWithCIDRAllocator(test.CloudQPS, clusterCIDR, subnetMaskSize)
			case ipam.IPAMFromClusterAllocatorType:
				cs = newCloudServer(test.CloudQPS)
			}
			if cs != nil {
				stopCloud, cloud, err := util.NewMockGCECloud(cs)
				if err != nil {
					t.Errorf("Error creating test infrastructure: %v", err)
					return
				}
				test.Cloud = cloud
				defer stopCloud()
			}
			if results, err := runTest(t, apiURL, test, clusterCIDR, serviceCIDR, subnetMaskSize); err == nil {
				allResults = append(allResults, results)
			}
		})
	}

	logResults(allResults)
}

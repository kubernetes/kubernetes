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

package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"

	"k8s.io/klog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"

	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/ingress"
	"k8s.io/kubernetes/test/e2e/framework/providers/gce"
	"k8s.io/kubernetes/test/e2e/network/scale"
)

var (
	kubeconfig       string
	enableTLS        bool
	numIngressesTest numIngressesSlice
	testNamespace    string
	cloudConfig      framework.CloudConfig
	outputFile       string
	cleanup          bool
)

type numIngressesSlice []int

func (i *numIngressesSlice) String() string {
	return fmt.Sprintf("%d", *i)
}

func (i *numIngressesSlice) Set(value string) error {
	v, err := strconv.Atoi(value)
	if err != nil {
		return err
	}
	*i = append(*i, v)
	sort.Ints(*i)
	return nil
}

func registerFlags() {
	if home := os.Getenv("HOME"); home != "" {
		flag.StringVar(&kubeconfig, "kubeconfig", filepath.Join(home, ".kube", "config"), "(optional) Absolute path to the kubeconfig file")
	} else {
		flag.StringVar(&kubeconfig, "kubeconfig", "", "Absolute path to the kubeconfig file")
	}
	flag.StringVar(&cloudConfig.ProjectID, "project", "", "GCE project being used")
	flag.StringVar(&cloudConfig.Zone, "zone", "", "GCE zone being used")
	flag.StringVar(&cloudConfig.Region, "region", "", "GCE region being used")
	flag.Var(&numIngressesTest, "num-ingresses", "The number of ingresses to test, specify multiple times for step testing (e.g. 5 ingresses -> 20 ingresses -> 100 ingresses)")
	flag.BoolVar(&enableTLS, "enable-tls", true, "Whether to enable TLS on ingress")
	flag.StringVar(&testNamespace, "namespace", "ingress-test-scale", "Namespace for testing")
	flag.StringVar(&outputFile, "output", "", "If specify, dump latencies to the specified file")
	flag.BoolVar(&cleanup, "cleanup", true, "Whether to cleanup resources after test")
}

func verifyFlags() error {
	if cloudConfig.ProjectID == "" || cloudConfig.Zone == "" || cloudConfig.Region == "" {
		return fmt.Errorf("must set all of --project, --zone and --region")
	}
	return nil
}

func main() {
	registerFlags()
	flag.Parse()
	if err := verifyFlags(); err != nil {
		klog.Errorf("Failed to verify flags: %v", err)
		os.Exit(1)
	}

	// Initializing a k8s client.
	config, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		klog.Errorf("Failed to build kubeconfig: %v", err)
		os.Exit(1)
	}
	cs, err := clientset.NewForConfig(config)
	if err != nil {
		klog.Errorf("Failed to create kubeclient: %v", err)
		os.Exit(1)
	}

	// Initializing a GCE client.
	gceAlphaFeatureGate := gcecloud.NewAlphaFeatureGate([]string{})
	gceCloud, err := gcecloud.CreateGCECloud(&gcecloud.CloudConfig{
		ProjectID:        cloudConfig.ProjectID,
		Region:           cloudConfig.Region,
		Zone:             cloudConfig.Zone,
		AlphaFeatureGate: gceAlphaFeatureGate,
	})
	if err != nil {
		klog.Errorf("Error building GCE provider: %v", err)
		os.Exit(1)
	}
	cloudConfig.Provider = gce.NewProvider(gceCloud)

	testSuccessFlag := true
	defer func() {
		if !testSuccessFlag {
			klog.Errorf("Ingress scale test failed.")
			os.Exit(1)
		}
	}()

	ns := &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: testNamespace,
		},
	}
	klog.Infof("Creating namespace %s...", ns.Name)
	if _, err := cs.CoreV1().Namespaces().Create(ns); err != nil {
		klog.Errorf("Failed to create namespace %s: %v", ns.Name, err)
		testSuccessFlag = false
		return
	}
	if cleanup {
		defer func() {
			klog.Infof("Deleting namespace %s...", ns.Name)
			if err := cs.CoreV1().Namespaces().Delete(ns.Name, nil); err != nil {
				klog.Errorf("Failed to delete namespace %s: %v", ns.Name, err)
				testSuccessFlag = false
			}
		}()
	}

	// Setting up a localized scale test framework.
	f := scale.NewIngressScaleFramework(cs, ns.Name, cloudConfig)
	f.Logger = &ingress.GLogger{}
	// Customizing scale test.
	f.EnableTLS = enableTLS
	f.OutputFile = outputFile
	if len(numIngressesTest) != 0 {
		f.NumIngressesTest = numIngressesTest
	}

	// Real test begins.
	if cleanup {
		defer func() {
			if errs := f.CleanupScaleTest(); len(errs) != 0 {
				klog.Errorf("Failed to cleanup scale test: %v", errs)
				testSuccessFlag = false
			}
		}()
	}
	err = f.PrepareScaleTest()
	if err != nil {
		klog.Errorf("Failed to prepare scale test: %v", err)
		testSuccessFlag = false
		return
	}

	if errs := f.RunScaleTest(); len(errs) != 0 {
		klog.Errorf("Failed while running scale test: %v", errs)
		testSuccessFlag = false
	}
}

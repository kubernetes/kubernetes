/*
Copyright 2022 The Kubernetes Authors.

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

package servicecidr

import (
	"context"
	"fmt"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
)

func TestServiceCIDR(t *testing.T) {
	// set the feature gate to enable MultiCIDRRangeAllocator
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.MultiCIDRServiceAllocator, true)()

	serviceCIDR := "10.12.0.0/16"
	secondaryServiceCIDR := "2001:db8:1::/112"

	_, kubeConfig, tearDownFn := framework.StartTestServer(t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			// Disable ServiceAccount admission plugin as we don't have serviceaccount controller running.
			opts.Admission.GenericAdmission.DisablePlugins = []string{"ServiceAccount", "TaintNodesByCondition"}
			opts.APIEnablement.RuntimeConfig.Set("networking.k8s.io/v1alpha1=true")
			opts.ServiceClusterIPRanges = fmt.Sprintf("%s,%s", serviceCIDR, secondaryServiceCIDR)
		},
	})
	defer tearDownFn()

	client := clientset.NewForConfigOrDie(kubeConfig)

	ns := framework.CreateNamespaceOrDie(client, "test-service-cidr", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sharedInformer := informers.NewSharedInformerFactory(client, 1*time.Hour)
	sharedInformer.Start(ctx.Done())

	ipPolicy := v1.IPFamilyPolicyRequireDualStack
	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-123",
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{
				{
					Port:       443,
					TargetPort: intstr.FromInt(443),
				},
			},
			IPFamilyPolicy: &ipPolicy,
		},
	}

	service, err := client.CoreV1().Services(ns.Name).Create(context.TODO(), service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}
	service, err = client.CoreV1().Services(ns.Name).Get(context.TODO(), service.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("error getting service: %v", err)
	}
	fmt.Println("----------- Service ", service)

	svcList, err := client.NetworkingV1alpha1().ServiceCIDRs().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Error(err)
	}
	fmt.Println("----------- Service CIDRs", svcList)
	ipList, err := client.NetworkingV1alpha1().IPAddresses().List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		t.Error(err)
	}
	fmt.Println("----------- Service IP Addresses", ipList)

}

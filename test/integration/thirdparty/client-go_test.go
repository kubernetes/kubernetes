/*
Copyright 2017 The Kubernetes Authors.

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

package thirdparty

import (
	"testing"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/pkg/api"
	"k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/integration/framework"

	exampletprv1 "k8s.io/client-go/examples/third-party-resources/apis/tpr/v1"
	exampleclient "k8s.io/client-go/examples/third-party-resources/client"
)

func TestClientGoThirdPartyResourceExample(t *testing.T) {
	_, s := framework.RunAMaster(framework.NewIntegrationTestMasterConfig())
	defer s.Close()

	config := &rest.Config{Host: s.URL, ContentConfig: rest.ContentConfig{NegotiatedSerializer: api.Codecs}}
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Creating TPR %q", exampletprv1.ExampleResourcePlural)
	if err := exampleclient.CreateTPR(clientset); err != nil {
		t.Fatalf("unexpected error creating the ThirdPartyResource: %v", err)
	}

	exampleClient, _, err := exampleclient.NewClient(config)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Waiting for TPR %q to show up", exampletprv1.ExampleResourcePlural)
	if err := exampleclient.WaitForExampleResource(exampleClient); err != nil {
		t.Fatalf("ThirdPartyResource examples did not show up: %v", err)
	}
	t.Logf("TPR %q is active", exampletprv1.ExampleResourcePlural)

}

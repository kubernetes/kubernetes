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

package monitoring

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	instrumentation "k8s.io/kubernetes/test/e2e/instrumentation/common"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"golang.org/x/oauth2/google"
)

const (
	// Time to wait after a pod creation for it's metadata to be exported
	metadataWaitTime = 120 * time.Second

	// MonitoringScope is the scope for Stackdriver Metadata API
	MonitoringScope = "https://www.googleapis.com/auth/monitoring"
)

var _ = instrumentation.SIGDescribe("Stackdriver Monitoring", func() {
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
	})

	f := framework.NewDefaultFramework("stackdriver-monitoring")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	var kubeClient clientset.Interface

	f.It("should run Stackdriver Metadata Agent", feature.StackdriverMetadataAgent, func(ctx context.Context) {
		kubeClient = f.ClientSet
		testAgent(ctx, f, kubeClient)
	})
})

func testAgent(ctx context.Context, f *framework.Framework, kubeClient clientset.Interface) {
	projectID := framework.TestContext.CloudConfig.ProjectID
	resourceType := "k8s_container"
	uniqueContainerName := fmt.Sprintf("test-container-%v", time.Now().Unix())
	endpoint := fmt.Sprintf(
		"https://stackdriver.googleapis.com/v1beta2/projects/%v/resourceMetadata?filter=resource.type%%3D%v+AND+resource.label.container_name%%3D%v",
		projectID,
		resourceType,
		uniqueContainerName)

	oauthClient, err := google.DefaultClient(ctx, MonitoringScope)
	if err != nil {
		framework.Failf("Failed to create oauth client: %s", err)
	}

	// Create test pod with unique name.
	_ = e2epod.CreateExecPodOrFail(ctx, kubeClient, f.Namespace.Name, uniqueContainerName, func(pod *v1.Pod) {
		pod.Spec.Containers[0].Name = uniqueContainerName
	})
	ginkgo.DeferCleanup(kubeClient.CoreV1().Pods(f.Namespace.Name).Delete, uniqueContainerName, metav1.DeleteOptions{})

	// Wait a short amount of time for Metadata Agent to be created and metadata to be exported
	time.Sleep(metadataWaitTime)

	resp, err := oauthClient.Get(endpoint)
	if err != nil {
		framework.Failf("Failed to call Stackdriver Metadata API %s", err)
	}
	if resp.StatusCode != 200 {
		framework.Failf("Stackdriver Metadata API returned error status: %s", resp.Status)
	}
	metadataAPIResponse, err := io.ReadAll(resp.Body)
	if err != nil {
		framework.Failf("Failed to read response from Stackdriver Metadata API: %s", err)
	}

	exists, err := verifyPodExists(metadataAPIResponse, uniqueContainerName)
	if err != nil {
		framework.Failf("Failed to process response from Stackdriver Metadata API: %s", err)
	}
	if !exists {
		framework.Failf("Missing Metadata for container %q", uniqueContainerName)
	}
}

// Metadata has the information fetched from Stackdriver metadata API.
type Metadata struct {
	Results []map[string]interface{}
}

// Resource contains the resource type and labels from Stackdriver metadata API.
type Resource struct {
	resourceType   string
	resourceLabels map[string]string
}

func verifyPodExists(response []byte, containerName string) (bool, error) {
	var metadata Metadata
	err := json.Unmarshal(response, &metadata)
	if err != nil {
		return false, fmt.Errorf("Failed to unmarshall: %w", err)
	}

	for _, result := range metadata.Results {
		rawResource, ok := result["resource"]
		if !ok {
			return false, fmt.Errorf("No resource entry in response from Stackdriver Metadata API")
		}
		resource, err := parseResource(rawResource)
		if err != nil {
			return false, fmt.Errorf("No 'resource' label: %w", err)
		}
		if resource.resourceType == "k8s_container" &&
			resource.resourceLabels["container_name"] == containerName {
			return true, nil
		}
	}
	return false, nil
}

func parseResource(resource interface{}) (*Resource, error) {
	labels := map[string]string{}
	resourceMap, ok := resource.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("Resource entry is of type %s, expected map[string]interface{}", reflect.TypeOf(resource))
	}
	resourceType, ok := resourceMap["type"]
	if !ok {
		return nil, fmt.Errorf("Resource entry doesn't have a type specified")
	}
	resourceTypeName, ok := resourceType.(string)
	if !ok {
		return nil, fmt.Errorf("Resource type is of type %s, expected string", reflect.TypeOf(resourceType))
	}
	resourceLabels, ok := resourceMap["labels"]
	if !ok {
		return nil, fmt.Errorf("Resource entry doesn't have any labels specified")
	}
	resourceLabelMap, ok := resourceLabels.(map[string]interface{})
	if !ok {
		return nil, fmt.Errorf("Resource labels entry is of type %s, expected map[string]interface{}", reflect.TypeOf(resourceLabels))
	}
	for label, val := range resourceLabelMap {
		labels[label], ok = val.(string)
		if !ok {
			return nil, fmt.Errorf("Resource label %q is of type %s, expected string", label, reflect.TypeOf(val))
		}
	}
	return &Resource{
		resourceType:   resourceTypeName,
		resourceLabels: labels,
	}, nil
}

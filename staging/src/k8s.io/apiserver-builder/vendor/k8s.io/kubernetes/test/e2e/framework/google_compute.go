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

package framework

import (
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
)

// TODO: These should really just use the GCE API client library or at least use
// better formatted output from the --format flag.

func CreateGCEStaticIP(name string) (string, error) {
	// gcloud compute --project "abshah-kubernetes-001" addresses create "test-static-ip" --region "us-central1"
	// abshah@abhidesk:~/go/src/code.google.com/p/google-api-go-client/compute/v1$ gcloud compute --project "abshah-kubernetes-001" addresses create "test-static-ip" --region "us-central1"
	// Created [https://www.googleapis.com/compute/v1/projects/abshah-kubernetes-001/regions/us-central1/addresses/test-static-ip].
	// NAME           REGION      ADDRESS       STATUS
	// test-static-ip us-central1 104.197.143.7 RESERVED

	var outputBytes []byte
	var err error
	region, err := gce.GetGCERegion(TestContext.CloudConfig.Zone)
	if err != nil {
		return "", fmt.Errorf("failed to convert zone to region: %v", err)
	}
	glog.Infof("Creating static IP with name %q in project %q in region %q", name, TestContext.CloudConfig.ProjectID, region)
	for attempts := 0; attempts < 4; attempts++ {
		outputBytes, err = exec.Command("gcloud", "compute", "addresses", "create",
			name, "--project", TestContext.CloudConfig.ProjectID,
			"--region", region, "-q").CombinedOutput()
		if err == nil {
			break
		}
		glog.Errorf("output from failed attempt to create static IP: %s", outputBytes)
		time.Sleep(time.Duration(5*attempts) * time.Second)
	}
	if err != nil {
		// Ditch the error, since the stderr in the output is what actually contains
		// any useful info.
		return "", fmt.Errorf("failed to create static IP: %s", outputBytes)
	}
	output := string(outputBytes)
	if strings.Contains(output, "RESERVED") {
		r, _ := regexp.Compile("[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+")
		staticIP := r.FindString(output)
		if staticIP == "" {
			return "", fmt.Errorf("static IP not found in gcloud command output: %v", output)
		} else {
			return staticIP, nil
		}
	} else {
		return "", fmt.Errorf("static IP %q could not be reserved: %v", name, output)
	}
}

func DeleteGCEStaticIP(name string) error {
	// gcloud compute --project "abshah-kubernetes-001" addresses create "test-static-ip" --region "us-central1"
	// abshah@abhidesk:~/go/src/code.google.com/p/google-api-go-client/compute/v1$ gcloud compute --project "abshah-kubernetes-001" addresses create "test-static-ip" --region "us-central1"
	// Created [https://www.googleapis.com/compute/v1/projects/abshah-kubernetes-001/regions/us-central1/addresses/test-static-ip].
	// NAME           REGION      ADDRESS       STATUS
	// test-static-ip us-central1 104.197.143.7 RESERVED

	region, err := gce.GetGCERegion(TestContext.CloudConfig.Zone)
	if err != nil {
		return fmt.Errorf("failed to convert zone to region: %v", err)
	}
	glog.Infof("Deleting static IP with name %q in project %q in region %q", name, TestContext.CloudConfig.ProjectID, region)
	outputBytes, err := exec.Command("gcloud", "compute", "addresses", "delete",
		name, "--project", TestContext.CloudConfig.ProjectID,
		"--region", region, "-q").CombinedOutput()
	if err != nil {
		// Ditch the error, since the stderr in the output is what actually contains
		// any useful info.
		return fmt.Errorf("failed to delete static IP %q: %v", name, string(outputBytes))
	}
	return nil
}

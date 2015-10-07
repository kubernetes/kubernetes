/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package e2e

import (
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"github.com/golang/glog"
)

func createGCEStaticIP(name string) (string, error) {
	// gcloud compute --project "abshah-kubernetes-001" addresses create "test-static-ip" --region "us-central1"
	// abshah@abhidesk:~/go/src/code.google.com/p/google-api-go-client/compute/v1$ gcloud compute --project "abshah-kubernetes-001" addresses create "test-static-ip" --region "us-central1"
	// Created [https://www.googleapis.com/compute/v1/projects/abshah-kubernetes-001/regions/us-central1/addresses/test-static-ip].
	// NAME           REGION      ADDRESS       STATUS
	// test-static-ip us-central1 104.197.143.7 RESERVED

	var output []byte
	var err error
	for attempts := 0; attempts < 4; attempts++ {
		output, err = exec.Command("gcloud", "compute", "addresses", "create",
			name, "--project", testContext.CloudConfig.ProjectID,
			"--region", "us-central1", "-q").CombinedOutput()
		if err == nil {
			break
		}
		glog.Errorf("Creating static IP with name:%s in project: %s", name, testContext.CloudConfig.ProjectID)
		glog.Errorf("output: %s", output)
		time.Sleep(time.Duration(5*attempts) * time.Second)
	}
	if err != nil {
		return "", err
	}
	text := string(output)
	if strings.Contains(text, "RESERVED") {
		r, _ := regexp.Compile("[0-9]+\\.[0-9]+\\.[0-9]+\\.[0-9]+")
		staticIP := r.FindString(text)
		if staticIP == "" {
			glog.Errorf("Static IP creation output is \n %s", text)
			return "", fmt.Errorf("Static IP not found in gcloud compute command output")
		} else {
			return staticIP, nil
		}
	} else {
		return "", fmt.Errorf("Static IP Could not be reserved.")
	}
}

func deleteGCEStaticIP(name string) error {
	// gcloud compute --project "abshah-kubernetes-001" addresses create "test-static-ip" --region "us-central1"
	// abshah@abhidesk:~/go/src/code.google.com/p/google-api-go-client/compute/v1$ gcloud compute --project "abshah-kubernetes-001" addresses create "test-static-ip" --region "us-central1"
	// Created [https://www.googleapis.com/compute/v1/projects/abshah-kubernetes-001/regions/us-central1/addresses/test-static-ip].
	// NAME           REGION      ADDRESS       STATUS
	// test-static-ip us-central1 104.197.143.7 RESERVED

	_, err := exec.Command("gcloud", "compute", "addresses", "delete",
		name, "--project", testContext.CloudConfig.ProjectID,
		"--region", "us-central1", "-q").CombinedOutput()
	return err
}

/*
Copyright 2016 The Kubernetes Authors.

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

package common

import (
	"fmt"
	"os/exec"
	"regexp"
	"time"

	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/autoscaling"

	"k8s.io/apimachinery/pkg/util/sets"
	awscloud "k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
	"k8s.io/kubernetes/test/e2e/framework"
)

type Suite string

const (
	E2E           Suite = "e2e"
	NodeE2E       Suite = "node e2e"
	FederationE2E Suite = "federation e2e"
)

var CurrentSuite Suite

// CommonImageWhiteList is the list of images used in common test. These images should be prepulled
// before a tests starts, so that the tests won't fail due image pulling flakes. Currently, this is
// only used by node e2e test.
// TODO(random-liu): Change the image puller pod to use similar mechanism.
var CommonImageWhiteList = sets.NewString(
	"gcr.io/google_containers/busybox:1.24",
	"gcr.io/google_containers/eptest:0.1",
	"gcr.io/google_containers/liveness:e2e",
	"gcr.io/google_containers/mounttest:0.8",
	"gcr.io/google_containers/mounttest-user:0.5",
	"gcr.io/google_containers/netexec:1.4",
	"gcr.io/google_containers/netexec:1.5",
	"gcr.io/google_containers/nginx-slim:0.7",
	"gcr.io/google_containers/serve_hostname:v1.4",
	"gcr.io/google_containers/test-webserver:e2e",
	"gcr.io/google_containers/hostexec:1.2",
	"gcr.io/google_containers/volume-nfs:0.8",
	"gcr.io/google_containers/volume-gluster:0.2",
)

// TODO: This is a temporary home, it needs to find a clean landing location
func GroupSize(group string) (int, error) {
	if framework.TestContext.Provider == "gce" || framework.TestContext.Provider == "gke" {
		// TODO: make this hit the compute API directly instead of shelling out to gcloud.
		// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed",
			"list-instances", group, "--project="+framework.TestContext.CloudConfig.ProjectID,
			"--zone="+framework.TestContext.CloudConfig.Zone).CombinedOutput()
		if err != nil {
			return -1, err
		}
		re := regexp.MustCompile("RUNNING")
		return len(re.FindAllString(string(output), -1)), nil
	} else if framework.TestContext.Provider == "aws" {
		client := autoscaling.New(session.New())
		instanceGroup, err := awscloud.DescribeInstanceGroup(client, group)
		if err != nil {
			return -1, fmt.Errorf("error describing instance group: %v", err)
		}
		if instanceGroup == nil {
			return -1, fmt.Errorf("instance group not found: %s", group)
		}
		return instanceGroup.CurrentSize()
	} else {
		return -1, fmt.Errorf("provider does not support InstanceGroups")
	}
}

// TODO: This is a temporary home, it needs to find a clean landing location
func WaitForGroupSize(group string, size int32) error {
	timeout := 30 * time.Minute
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		currentSize, err := GroupSize(group)
		if err != nil {
			framework.Logf("Failed to get node instance group size: %v", err)
			continue
		}
		if currentSize != int(size) {
			framework.Logf("Waiting for node instance group size %d, current size %d", size, currentSize)
			continue
		}
		framework.Logf("Node instance group has reached the desired size %d", size)
		return nil
	}
	return fmt.Errorf("timeout waiting %v for node instance group size to be %d", timeout, size)
}

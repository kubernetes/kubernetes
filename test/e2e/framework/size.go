/*
Copyright 2015 The Kubernetes Authors.

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

	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/autoscaling"
	awscloud "k8s.io/kubernetes/pkg/cloudprovider/providers/aws"
)

const (
	resizeNodeReadyTimeout    = 2 * time.Minute
	resizeNodeNotReadyTimeout = 2 * time.Minute
)

func ResizeGroup(group string, size int32) error {
	if TestContext.ReportDir != "" {
		CoreDump(TestContext.ReportDir)
		defer CoreDump(TestContext.ReportDir)
	}
	if TestContext.Provider == "gce" || TestContext.Provider == "gke" {
		// TODO: make this hit the compute API directly instead of shelling out to gcloud.
		// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed", "resize",
			group, fmt.Sprintf("--size=%v", size),
			"--project="+TestContext.CloudConfig.ProjectID, "--zone="+TestContext.CloudConfig.Zone).CombinedOutput()
		if err != nil {
			Logf("Failed to resize node instance group: %v", string(output))
		}
		return err
	} else if TestContext.Provider == "aws" {
		client := autoscaling.New(session.New())
		return awscloud.ResizeInstanceGroup(client, group, int(size))
	} else if TestContext.Provider == "kubemark" {
		return TestContext.CloudConfig.KubemarkController.SetNodeGroupSize(group, int(size))
	} else {
		return fmt.Errorf("Provider does not support InstanceGroups")
	}
}

func GetGroupNodes(group string) ([]string, error) {
	if TestContext.Provider == "gce" || TestContext.Provider == "gke" {
		// TODO: make this hit the compute API directly instead of shelling out to gcloud.
		// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed",
			"list-instances", group, "--project="+TestContext.CloudConfig.ProjectID,
			"--zone="+TestContext.CloudConfig.Zone).CombinedOutput()
		if err != nil {
			Logf("Failed to get nodes in instance group: %v", string(output))
			return nil, err
		}
		re := regexp.MustCompile(".*RUNNING")
		lines := re.FindAllString(string(output), -1)
		for i, line := range lines {
			lines[i] = line[:strings.Index(line, " ")]
		}
		return lines, nil
	} else if TestContext.Provider == "kubemark" {
		return TestContext.CloudConfig.KubemarkController.GetNodeNamesForNodeGroup(group)
	} else {
		return nil, fmt.Errorf("provider does not support InstanceGroups")
	}
}

func GroupSize(group string) (int, error) {
	if TestContext.Provider == "gce" || TestContext.Provider == "gke" {
		// TODO: make this hit the compute API directly instead of shelling out to gcloud.
		// TODO: make gce/gke implement InstanceGroups, so we can eliminate the per-provider logic
		output, err := exec.Command("gcloud", "compute", "instance-groups", "managed",
			"list-instances", group, "--project="+TestContext.CloudConfig.ProjectID,
			"--zone="+TestContext.CloudConfig.Zone).CombinedOutput()
		if err != nil {
			return -1, err
		}
		re := regexp.MustCompile("RUNNING")
		return len(re.FindAllString(string(output), -1)), nil
	} else if TestContext.Provider == "aws" {
		client := autoscaling.New(session.New())
		instanceGroup, err := awscloud.DescribeInstanceGroup(client, group)
		if err != nil {
			return -1, fmt.Errorf("error describing instance group: %v", err)
		}
		if instanceGroup == nil {
			return -1, fmt.Errorf("instance group not found: %s", group)
		}
		return instanceGroup.CurrentSize()
	} else if TestContext.Provider == "kubemark" {
		return TestContext.CloudConfig.KubemarkController.GetNodeGroupSize(group)
	} else {
		return -1, fmt.Errorf("provider does not support InstanceGroups")
	}
}

func WaitForGroupSize(group string, size int32) error {
	timeout := 30 * time.Minute
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(20 * time.Second) {
		currentSize, err := GroupSize(group)
		if err != nil {
			Logf("Failed to get node instance group size: %v", err)
			continue
		}
		if currentSize != int(size) {
			Logf("Waiting for node instance group size %d, current size %d", size, currentSize)
			continue
		}
		Logf("Node instance group has reached the desired size %d", size)
		return nil
	}
	return fmt.Errorf("timeout waiting %v for node instance group size to be %d", timeout, size)
}

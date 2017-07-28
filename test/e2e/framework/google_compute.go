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
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os/exec"
	"path/filepath"
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
			"--region", region, "-q", "--format=yaml").CombinedOutput()
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

// Returns master & node image string, or error
func lookupClusterImageSources() (string, string, error) {
	// Given args for a gcloud compute command, run it with other args, and return the values,
	// whether separated by newlines, commas or semicolons.
	gcloudf := func(argv ...string) ([]string, error) {
		args := []string{"compute"}
		args = append(args, argv...)
		args = append(args, "--project", TestContext.CloudConfig.ProjectID,
			"--zone", TestContext.CloudConfig.Zone)
		outputBytes, err := exec.Command("gcloud", args...).CombinedOutput()
		str := strings.Replace(string(outputBytes), ",", "\n", -1)
		str = strings.Replace(str, ";", "\n", -1)
		lines := strings.Split(str, "\n")
		if err != nil {
			Logf("lookupDiskImageSources: gcloud error with [%#v]; err:%v", argv, err)
			for _, l := range lines {
				Logf(" > %s", l)
			}
		}
		return lines, err
	}

	// Given a GCE instance, look through its disks, finding one that has a sourceImage
	host2image := func(instance string) (string, error) {
		// gcloud compute instances describe {INSTANCE} --format="get(disks[].source)"
		// gcloud compute disks describe {DISKURL} --format="get(sourceImage)"
		disks, err := gcloudf("instances", "describe", instance, "--format=get(disks[].source)")
		if err != nil {
			return "", err
		} else if len(disks) == 0 {
			return "", fmt.Errorf("instance %q had no findable disks", instance)
		}
		// Loop over disks, looking for the boot disk
		for _, disk := range disks {
			lines, err := gcloudf("disks", "describe", disk, "--format=get(sourceImage)")
			if err != nil {
				return "", err
			} else if len(lines) > 0 && lines[0] != "" {
				return lines[0], nil // break, we're done
			}
		}
		return "", fmt.Errorf("instance %q had no disk with a sourceImage", instance)
	}

	// gcloud compute instance-groups list-instances {GROUPNAME} --format="get(instance)"
	nodeName := ""
	instGroupName := strings.Split(TestContext.CloudConfig.NodeInstanceGroup, ",")[0]
	if lines, err := gcloudf("instance-groups", "list-instances", instGroupName, "--format=get(instance)"); err != nil {
		return "", "", err
	} else if len(lines) == 0 {
		return "", "", fmt.Errorf("no instances inside instance-group %q", instGroupName)
	} else {
		nodeName = lines[0]
	}

	nodeImg, err := host2image(nodeName)
	if err != nil {
		return "", "", err
	}
	frags := strings.Split(nodeImg, "/")
	nodeImg = frags[len(frags)-1]

	// For GKE clusters, MasterName will not be defined; we just leave masterImg blank.
	masterImg := ""
	if masterName := TestContext.CloudConfig.MasterName; masterName != "" {
		img, err := host2image(masterName)
		if err != nil {
			return "", "", err
		}
		frags = strings.Split(img, "/")
		masterImg = frags[len(frags)-1]
	}

	return masterImg, nodeImg, nil
}

func LogClusterImageSources() {
	masterImg, nodeImg, err := lookupClusterImageSources()
	if err != nil {
		Logf("Cluster image sources lookup failed: %v\n", err)
		return
	}
	Logf("cluster-master-image: %s", masterImg)
	Logf("cluster-node-image: %s", nodeImg)

	images := map[string]string{
		"master_os_image": masterImg,
		"node_os_image":   nodeImg,
	}

	outputBytes, _ := json.MarshalIndent(images, "", "  ")
	filePath := filepath.Join(TestContext.ReportDir, "images.json")
	if err := ioutil.WriteFile(filePath, outputBytes, 0644); err != nil {
		Logf("cluster images sources, could not write to %q: %v", filePath, err)
	}
}

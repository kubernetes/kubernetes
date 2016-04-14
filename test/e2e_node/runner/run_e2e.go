/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// To run the e2e tests against one or more hosts on gce:
// $ go run run_e2e.go --logtostderr --v 2 --ssh-env gce --hosts <comma separated hosts>
// To run the e2e tests against one or more images on gce and provision them:
// $ go run run_e2e.go --logtostderr --v 2 --project <project> --zone <zone> --ssh-env gce --images <comma separated images>
package main

import (
	"flag"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"strings"
	"time"

	"k8s.io/kubernetes/test/e2e_node"

	"github.com/golang/glog"
	"github.com/pborman/uuid"
	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/compute/v1"
)

var instanceNamePrefix = flag.String("instance-name-prefix", "", "prefix for instance names")
var zone = flag.String("zone", "", "gce zone the hosts live in")
var project = flag.String("project", "", "gce project the hosts live in")
var images = flag.String("images", "", "images to test")
var hosts = flag.String("hosts", "", "hosts to test")
var cleanup = flag.Bool("cleanup", true, "If true remove files from remote hosts and delete temporary instances")
var buildOnly = flag.Bool("build-only", false, "If true, build e2e_node_test.tar.gz and exit.")

var computeService *compute.Service

type TestResult struct {
	output string
	err    error
	host   string
}

func main() {
	flag.Parse()
	rand.Seed(time.Now().UTC().UnixNano())
	if *buildOnly {
		// Build the archive and exit
		e2e_node.CreateTestArchive()
		return
	}

	if *hosts == "" && *images == "" {
		glog.Fatalf("Must specify one of --images or --hosts flag.")
	}
	if *images != "" && *zone == "" {
		glog.Fatal("Must specify --zone flag")
	}
	if *images != "" && *project == "" {
		glog.Fatal("Must specify --project flag")
	}
	if *instanceNamePrefix == "" {
		*instanceNamePrefix = "tmp-node-e2e-" + uuid.NewUUID().String()[:8]
	}

	// Setup coloring
	stat, _ := os.Stdout.Stat()
	useColor := (stat.Mode() & os.ModeCharDevice) != 0
	blue := ""
	noColour := ""
	if useColor {
		blue = "\033[0;34m"
		noColour = "\033[0m"
	}

	archive := e2e_node.CreateTestArchive()
	defer os.Remove(archive)

	results := make(chan *TestResult)
	running := 0
	if *images != "" {
		// Setup the gce client for provisioning instances
		// Getting credentials on gce jenkins is flaky, so try a couple times
		var err error
		for i := 0; i < 10; i++ {
			var client *http.Client
			client, err = google.DefaultClient(oauth2.NoContext, compute.ComputeScope)
			if err != nil {
				continue
			}
			computeService, err = compute.New(client)
			if err != nil {
				continue
			}
			time.Sleep(time.Second * 6)
		}
		if err != nil {
			glog.Fatalf("Unable to create gcloud compute service using defaults.  Make sure you are authenticated. %v", err)
		}

		for _, image := range strings.Split(*images, ",") {
			running++
			fmt.Printf("Initializing e2e tests using image %s.\n", image)
			go func(image string) { results <- testImage(image, archive) }(image)
		}
	}
	if *hosts != "" {
		for _, host := range strings.Split(*hosts, ",") {
			fmt.Printf("Initializing e2e tests using host %s.\n", host)
			running++
			go func(host string) {
				results <- testHost(host, archive, *cleanup)
			}(host)
		}
	}

	// Wait for all tests to complete and emit the results
	errCount := 0
	for i := 0; i < running; i++ {
		tr := <-results
		host := tr.host
		fmt.Printf("%s================================================================%s\n", blue, noColour)
		if tr.err != nil {
			errCount++
			fmt.Printf("Failure Finished Host %s Test Suite\n%s\n%v\n", host, tr.output, tr.err)
		} else {
			fmt.Printf("Success Finished Host %s Test Suite\n%s\n", host, tr.output)
		}
		fmt.Printf("%s================================================================%s\n", blue, noColour)
	}

	// Set the exit code if there were failures
	if errCount > 0 {
		fmt.Printf("Failure: %d errors encountered.", errCount)
		os.Exit(1)
	}
}

// Run tests in archive against host
func testHost(host, archive string, deleteFiles bool) *TestResult {
	output, err := e2e_node.RunRemote(archive, host, deleteFiles)
	return &TestResult{
		output: output,
		err:    err,
		host:   host,
	}
}

// Provision a gce instance using image and run the tests in archive against the instance.
// Delete the instance afterward.
func testImage(image, archive string) *TestResult {
	host, err := createInstance(image)
	if *cleanup {
		defer deleteInstance(image)
	}
	if err != nil {
		return &TestResult{
			err: fmt.Errorf("Unable to create gce instance with running docker daemon for image %s.  %v", image, err),
		}
	}
	return testHost(host, archive, false)
}

// Provision a gce instance using image
func createInstance(image string) (string, error) {
	name := imageToInstanceName(image)
	i := &compute.Instance{
		Name:        name,
		MachineType: machineType(),
		NetworkInterfaces: []*compute.NetworkInterface{
			{
				AccessConfigs: []*compute.AccessConfig{
					{
						Type: "ONE_TO_ONE_NAT",
						Name: "External NAT",
					},
				}},
		},
		Disks: []*compute.AttachedDisk{
			{
				AutoDelete: true,
				Boot:       true,
				Type:       "PERSISTENT",
				InitializeParams: &compute.AttachedDiskInitializeParams{
					SourceImage: sourceImage(image),
				},
			},
		},
	}
	op, err := computeService.Instances.Insert(*project, *zone, i).Do()
	if err != nil {
		return "", err
	}
	if op.Error != nil {
		return "", fmt.Errorf("Could not create instance %s: %+v", name, op.Error)
	}

	instanceRunning := false
	for i := 0; i < 30 && !instanceRunning; i++ {
		if i > 0 {
			time.Sleep(time.Second * 20)
		}
		var instance *compute.Instance
		instance, err = computeService.Instances.Get(*project, *zone, name).Do()
		if err != nil {
			continue
		}
		if strings.ToUpper(instance.Status) != "RUNNING" {
			err = fmt.Errorf("Instance %s not in state RUNNING, was %s.", name, instance.Status)
			continue
		}
		var output string
		output, err = e2e_node.RunSshCommand("ssh", name, "--", "sudo", "docker", "version")
		if err != nil {
			err = fmt.Errorf("Instance %s not running docker daemon - Command failed: %s", name, output)
			continue
		}
		if !strings.Contains(output, "Server") {
			err = fmt.Errorf("Instance %s not running docker daemon - Server not found: %s", name, output)
			continue
		}
		instanceRunning = true
	}
	return name, err
}

func deleteInstance(image string) {
	_, err := computeService.Instances.Delete(*project, *zone, imageToInstanceName(image)).Do()
	if err != nil {
		glog.Infof("Error deleting instance %s", imageToInstanceName(image))
	}
}

func imageToInstanceName(image string) string {
	return *instanceNamePrefix + "-" + image
}

func sourceImage(image string) string {
	return fmt.Sprintf("projects/%s/global/images/%s", *project, image)
}

func machineType() string {
	return fmt.Sprintf("zones/%s/machineTypes/n1-standard-1", *zone)
}

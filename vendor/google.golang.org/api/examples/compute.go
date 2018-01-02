// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	compute "google.golang.org/api/compute/v1"
	"google.golang.org/api/googleapi"
)

func init() {
	scopes := strings.Join([]string{
		compute.DevstorageFullControlScope,
		compute.ComputeScope,
	}, " ")
	registerDemo("compute", scopes, computeMain)
}

func computeMain(client *http.Client, argv []string) {
	if len(argv) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: compute project_id instance_name (to start an instance)")
		return
	}

	service, err := compute.New(client)
	if err != nil {
		log.Fatalf("Unable to create Compute service: %v", err)
	}

	projectId := argv[0]
	instanceName := argv[1]

	prefix := "https://www.googleapis.com/compute/v1/projects/" + projectId
	imageURL := "https://www.googleapis.com/compute/v1/projects/debian-cloud/global/images/debian-7-wheezy-v20140606"
	zone := "us-central1-a"

	// Show the current images that are available.
	res, err := service.Images.List(projectId).Do()
	log.Printf("Got compute.Images.List, err: %#v, %v", res, err)

	instance := &compute.Instance{
		Name:        instanceName,
		Description: "compute sample instance",
		MachineType: prefix + "/zones/" + zone + "/machineTypes/n1-standard-1",
		Disks: []*compute.AttachedDisk{
			{
				AutoDelete: true,
				Boot:       true,
				Type:       "PERSISTENT",
				InitializeParams: &compute.AttachedDiskInitializeParams{
					DiskName:    "my-root-pd",
					SourceImage: imageURL,
				},
			},
		},
		NetworkInterfaces: []*compute.NetworkInterface{
			&compute.NetworkInterface{
				AccessConfigs: []*compute.AccessConfig{
					&compute.AccessConfig{
						Type: "ONE_TO_ONE_NAT",
						Name: "External NAT",
					},
				},
				Network: prefix + "/global/networks/default",
			},
		},
		ServiceAccounts: []*compute.ServiceAccount{
			{
				Email: "default",
				Scopes: []string{
					compute.DevstorageFullControlScope,
					compute.ComputeScope,
				},
			},
		},
	}

	op, err := service.Instances.Insert(projectId, zone, instance).Do()
	log.Printf("Got compute.Operation, err: %#v, %v", op, err)
	etag := op.Header.Get("Etag")
	log.Printf("Etag=%v", etag)

	inst, err := service.Instances.Get(projectId, zone, instanceName).IfNoneMatch(etag).Do()
	log.Printf("Got compute.Instance, err: %#v, %v", inst, err)
	if googleapi.IsNotModified(err) {
		log.Printf("Instance not modified since insert.")
	} else {
		log.Printf("Instance modified since insert.")
	}
}

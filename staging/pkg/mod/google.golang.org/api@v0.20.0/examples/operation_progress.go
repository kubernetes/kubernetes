// Copyright 2018 Google LLC. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"log"

	"golang.org/x/oauth2/google"
	compute "google.golang.org/api/compute/v1"
)

const (
	projectID   = "some-project-id"
	zone        = "some-zone"
	operationID = "some-operation-id"
)

func operationProgressMain() {
	ctx := context.Background()

	client, err := google.DefaultClient(ctx, compute.CloudPlatformScope)
	if err != nil {
		log.Fatal(err)
	}
	svc, err := compute.New(client)
	if err != nil {
		log.Fatal(err)
	}

	for {
		resp, err := svc.ZoneOperations.Get(projectID, zone, operationID).Do()
		if err != nil {
			log.Fatal(err)
		}
		// Note: the response Status may differ between APIs. The string values
		// checked here may need to be changed depending on the API.
		if resp.Status != "WORKING" && resp.Status != "QUEUED" {
			break
		}
	}
	log.Println("operation complete")
}

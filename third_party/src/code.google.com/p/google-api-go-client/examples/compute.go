package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	compute "code.google.com/p/google-api-go-client/compute/v1beta12"
)

func init() {
	registerDemo("compute", compute.ComputeScope, computeMain)
}

func computeMain(client *http.Client, argv []string) {
	if len(argv) != 2 {
		fmt.Fprintln(os.Stderr, "Usage: compute project_id instance_name (to start an instance)")
		return
	}

	service, _ := compute.New(client)
	projectId := argv[0]
	instanceName := argv[1]

	prefix := "https://www.googleapis.com/compute/v1beta12/projects/" + projectId
	instance := &compute.Instance{
		Name:        instanceName,
		Description: "compute sample instance",
		Zone:        prefix + "/zones/us-east-a",
		MachineType: prefix + "/machine-types/standard-2-cpu-ephemeral-disk",
		NetworkInterfaces: []*compute.NetworkInterface{
			&compute.NetworkInterface{
				AccessConfigs: []*compute.AccessConfig{
					&compute.AccessConfig{Type: "ONE_TO_ONE_NAT"},
				},
				Network: prefix + "/networks/default",
			},
		},
	}
	op, err := service.Instances.Insert(projectId, instance).Do()
	log.Printf("Got compute.Operation, err: %#v, %v", op, err)
}

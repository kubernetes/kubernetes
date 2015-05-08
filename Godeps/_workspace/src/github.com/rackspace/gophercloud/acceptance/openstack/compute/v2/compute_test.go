// +build acceptance common

package v2

import (
	"fmt"
	"os"
	"strings"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/acceptance/tools"
	"github.com/rackspace/gophercloud/openstack"
	"github.com/rackspace/gophercloud/openstack/compute/v2/servers"
)

func newClient() (*gophercloud.ServiceClient, error) {
	ao, err := openstack.AuthOptionsFromEnv()
	if err != nil {
		return nil, err
	}

	client, err := openstack.AuthenticatedClient(ao)
	if err != nil {
		return nil, err
	}

	return openstack.NewComputeV2(client, gophercloud.EndpointOpts{
		Region: os.Getenv("OS_REGION_NAME"),
	})
}

func waitForStatus(client *gophercloud.ServiceClient, server *servers.Server, status string) error {
	return tools.WaitFor(func() (bool, error) {
		latest, err := servers.Get(client, server.ID).Extract()
		if err != nil {
			return false, err
		}

		if latest.Status == status {
			// Success!
			return true, nil
		}

		return false, nil
	})
}

// ComputeChoices contains image and flavor selections for use by the acceptance tests.
type ComputeChoices struct {
	// ImageID contains the ID of a valid image.
	ImageID string

	// FlavorID contains the ID of a valid flavor.
	FlavorID string

	// FlavorIDResize contains the ID of a different flavor available on the same OpenStack installation, that is distinct
	// from FlavorID.
	FlavorIDResize string

	// NetworkName is the name of a network to launch the instance on.
	NetworkName string
}

// ComputeChoicesFromEnv populates a ComputeChoices struct from environment variables.
// If any required state is missing, an `error` will be returned that enumerates the missing properties.
func ComputeChoicesFromEnv() (*ComputeChoices, error) {
	imageID := os.Getenv("OS_IMAGE_ID")
	flavorID := os.Getenv("OS_FLAVOR_ID")
	flavorIDResize := os.Getenv("OS_FLAVOR_ID_RESIZE")
	networkName := os.Getenv("OS_NETWORK_NAME")

	missing := make([]string, 0, 3)
	if imageID == "" {
		missing = append(missing, "OS_IMAGE_ID")
	}
	if flavorID == "" {
		missing = append(missing, "OS_FLAVOR_ID")
	}
	if flavorIDResize == "" {
		missing = append(missing, "OS_FLAVOR_ID_RESIZE")
	}
	if networkName == "" {
		networkName = "public"
	}

	notDistinct := ""
	if flavorID == flavorIDResize {
		notDistinct = "OS_FLAVOR_ID and OS_FLAVOR_ID_RESIZE must be distinct."
	}

	if len(missing) > 0 || notDistinct != "" {
		text := "You're missing some important setup:\n"
		if len(missing) > 0 {
			text += " * These environment variables must be provided: " + strings.Join(missing, ", ") + "\n"
		}
		if notDistinct != "" {
			text += " * " + notDistinct + "\n"
		}

		return nil, fmt.Errorf(text)
	}

	return &ComputeChoices{ImageID: imageID, FlavorID: flavorID, FlavorIDResize: flavorIDResize, NetworkName: networkName}, nil
}

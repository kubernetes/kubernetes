// +build acceptance compute limits

package v2

import (
	"strings"
	"testing"

	"github.com/gophercloud/gophercloud/acceptance/clients"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/limits"
)

func TestLimits(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	limits, err := limits.Get(client, nil).Extract()
	if err != nil {
		t.Fatalf("Unable to get limits: %v", err)
	}

	t.Logf("Limits for scoped user:")
	t.Logf("%#v", limits)
}

func TestLimitsForTenant(t *testing.T) {
	client, err := clients.NewComputeV2Client()
	if err != nil {
		t.Fatalf("Unable to create a compute client: %v", err)
	}

	// I think this is the easiest way to get the tenant ID while being
	// agnostic to Identity v2 and v3.
	// Technically we're just returning the limits for ourselves, but it's
	// the fact that we're specifying a tenant ID that is important here.
	endpointParts := strings.Split(client.Endpoint, "/")
	tenantID := endpointParts[4]

	getOpts := limits.GetOpts{
		TenantID: tenantID,
	}

	limits, err := limits.Get(client, getOpts).Extract()
	if err != nil {
		t.Fatalf("Unable to get absolute limits: %v", err)
	}

	t.Logf("Limits for tenant %s:", tenantID)
	t.Logf("%#v", limits)
}

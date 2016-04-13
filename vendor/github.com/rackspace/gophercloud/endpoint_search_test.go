package gophercloud

import (
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
)

func TestApplyDefaultsToEndpointOpts(t *testing.T) {
	eo := EndpointOpts{Availability: AvailabilityPublic}
	eo.ApplyDefaults("compute")
	expected := EndpointOpts{Availability: AvailabilityPublic, Type: "compute"}
	th.CheckDeepEquals(t, expected, eo)

	eo = EndpointOpts{Type: "compute"}
	eo.ApplyDefaults("object-store")
	expected = EndpointOpts{Availability: AvailabilityPublic, Type: "compute"}
	th.CheckDeepEquals(t, expected, eo)
}

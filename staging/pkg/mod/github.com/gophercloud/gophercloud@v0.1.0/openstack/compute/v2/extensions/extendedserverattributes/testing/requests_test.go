package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/extendedserverattributes"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestServerWithUsageExt(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/servers/d650a0ce-17c3-497d-961a-43c4af80998a", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Accept", "application/json")

		fmt.Fprintf(w, ServerWithAttributesExtResult)
	})

	type serverAttributesExt struct {
		servers.Server
		extendedserverattributes.ServerAttributesExt
	}
	var serverWithAttributesExt serverAttributesExt

	result := servers.Get(fake.ServiceClient(), "d650a0ce-17c3-497d-961a-43c4af80998a")

	// Extract basic fields.
	err := servers.Get(fake.ServiceClient(), "d650a0ce-17c3-497d-961a-43c4af80998a").ExtractInto(&serverWithAttributesExt)
	th.AssertNoErr(t, err)

	// Extract additional fields.
	reservationID, err := extendedserverattributes.ExtractReservationID(result.Result)
	th.AssertNoErr(t, err)

	launchIndex, err := extendedserverattributes.ExtractLaunchIndex(result.Result)
	th.AssertNoErr(t, err)

	hostname, err := extendedserverattributes.ExtractHostname(result.Result)
	th.AssertNoErr(t, err)

	rootDeviceName, err := extendedserverattributes.ExtractRootDeviceName(result.Result)
	th.AssertNoErr(t, err)

	th.AssertEquals(t, serverWithAttributesExt.Host, "compute01")
	th.AssertEquals(t, serverWithAttributesExt.InstanceName, "instance-00000001")
	th.AssertEquals(t, serverWithAttributesExt.HypervisorHostname, "compute01")
	th.AssertEquals(t, reservationID, "r-ky9gim1l")
	th.AssertEquals(t, launchIndex, 0)
	th.AssertEquals(t, hostname, "test00")
	th.AssertEquals(t, rootDeviceName, "/dev/sda")
}

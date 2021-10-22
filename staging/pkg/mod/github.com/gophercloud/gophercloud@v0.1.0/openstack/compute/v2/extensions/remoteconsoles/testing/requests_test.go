package testing

import (
	"fmt"
	"net/http"
	"testing"

	"github.com/gophercloud/gophercloud/openstack/compute/v2/extensions/remoteconsoles"
	th "github.com/gophercloud/gophercloud/testhelper"
	fake "github.com/gophercloud/gophercloud/testhelper/client"
)

func TestCreate(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/servers/b16ba811-199d-4ffd-8839-ba96c1185a67/remote-consoles", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "POST")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)
		th.TestHeader(t, r, "Content-Type", "application/json")
		th.TestHeader(t, r, "Accept", "application/json")
		th.TestJSONRequest(t, r, RemoteConsoleCreateRequest)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, RemoteConsoleCreateResult)
	})

	opts := remoteconsoles.CreateOpts{
		Protocol: remoteconsoles.ConsoleProtocolVNC,
		Type:     remoteconsoles.ConsoleTypeNoVNC,
	}
	s, err := remoteconsoles.Create(fake.ServiceClient(), "b16ba811-199d-4ffd-8839-ba96c1185a67", opts).Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.Protocol, string(remoteconsoles.ConsoleProtocolVNC))
	th.AssertEquals(t, s.Type, string(remoteconsoles.ConsoleTypeNoVNC))
	th.AssertEquals(t, s.URL, "http://192.168.0.4:6080/vnc_auto.html?token=9a2372b9-6a0e-4f71-aca1-56020e6bb677")
}

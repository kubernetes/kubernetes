package testing

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	fake "github.com/gophercloud/gophercloud/openstack/networking/v2/common"
	"github.com/gophercloud/gophercloud/openstack/networking/v2/extensions/agents"
	"github.com/gophercloud/gophercloud/pagination"
	th "github.com/gophercloud/gophercloud/testhelper"
)

func TestList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/agents", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, AgentsListResult)
	})

	count := 0

	agents.List(fake.ServiceClient(), agents.ListOpts{}).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := agents.ExtractAgents(page)

		if err != nil {
			t.Errorf("Failed to extract agents: %v", err)
			return false, nil
		}

		expected := []agents.Agent{
			Agent1,
			Agent2,
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	if count != 1 {
		t.Errorf("Expected 1 page, got %d", count)
	}
}

func TestGet(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/v2.0/agents/43583cf5-472e-4dc8-af5b-6aed4c94ee3a", func(w http.ResponseWriter, r *http.Request) {
		th.TestMethod(t, r, "GET")
		th.TestHeader(t, r, "X-Auth-Token", fake.TokenID)

		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)

		fmt.Fprintf(w, AgentsGetResult)
	})

	s, err := agents.Get(fake.ServiceClient(), "43583cf5-472e-4dc8-af5b-6aed4c94ee3a").Extract()
	th.AssertNoErr(t, err)

	th.AssertEquals(t, s.ID, "43583cf5-472e-4dc8-af5b-6aed4c94ee3a")
	th.AssertEquals(t, s.Binary, "neutron-openvswitch-agent")
	th.AssertEquals(t, s.AdminStateUp, true)
	th.AssertEquals(t, s.Alive, true)
	th.AssertEquals(t, s.Topic, "N/A")
	th.AssertEquals(t, s.Host, "compute3")
	th.AssertEquals(t, s.AgentType, "Open vSwitch agent")
	th.AssertEquals(t, s.HeartbeatTimestamp, time.Date(2019, 1, 9, 11, 43, 01, 0, time.UTC))
	th.AssertEquals(t, s.StartedAt, time.Date(2018, 6, 26, 21, 46, 20, 0, time.UTC))
	th.AssertEquals(t, s.CreatedAt, time.Date(2017, 7, 26, 23, 2, 5, 0, time.UTC))
	th.AssertDeepEquals(t, s.Configurations, map[string]interface{}{
		"ovs_hybrid_plug":            false,
		"datapath_type":              "system",
		"vhostuser_socket_dir":       "/var/run/openvswitch",
		"log_agent_heartbeats":       false,
		"l2_population":              true,
		"enable_distributed_routing": false,
	})
}

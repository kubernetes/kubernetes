package servers

import (
	"fmt"
	"net/http"
	"testing"
	"time"

	th "github.com/rackspace/gophercloud/testhelper"
	"github.com/rackspace/gophercloud/testhelper/client"
)

func TestWaitForStatus(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	th.Mux.HandleFunc("/servers/4321", func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
		{
			"server": {
				"name": "the-server",
				"id": "4321",
				"status": "ACTIVE"
			}
		}`)
	})

	err := WaitForStatus(client.ServiceClient(), "4321", "ACTIVE", 0)
	if err == nil {
		t.Errorf("Expected error: 'Time Out in WaitFor'")
	}

	err = WaitForStatus(client.ServiceClient(), "4321", "ACTIVE", 3)
	th.CheckNoErr(t, err)
}

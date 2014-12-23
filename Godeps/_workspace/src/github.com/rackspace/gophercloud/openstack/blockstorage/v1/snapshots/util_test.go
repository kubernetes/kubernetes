package snapshots

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

	th.Mux.HandleFunc("/snapshots/1234", func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
		{
			"snapshot": {
				"display_name": "snapshot-001",
				"id": "1234",
				"status":"available"
			}
		}`)
	})

	err := WaitForStatus(client.ServiceClient(), "1234", "available", 0)
	if err == nil {
		t.Errorf("Expected error: 'Time Out in WaitFor'")
	}

	err = WaitForStatus(client.ServiceClient(), "1234", "available", 3)
	th.CheckNoErr(t, err)
}

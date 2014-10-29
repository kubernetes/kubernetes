package volumes

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

	th.Mux.HandleFunc("/volumes/1234", func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(3 * time.Second)
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		fmt.Fprintf(w, `
		{
			"volume": {
				"display_name": "vol-001",
				"id": "1234",
				"status":"available"
			}
		}`)
	})

	err := WaitForStatus(client.ServiceClient(), "1234", "available", 0)
	if err == nil {
		t.Errorf("Expected error: 'Time Out in WaitFor'")
	}

	err = WaitForStatus(client.ServiceClient(), "1234", "available", 6)
	th.CheckNoErr(t, err)
}

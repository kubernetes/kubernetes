package roles

import (
	"testing"

	"github.com/rackspace/gophercloud"
)

func TestListAssignmentsURL(t *testing.T) {
	client := gophercloud.ServiceClient{Endpoint: "http://localhost:5000/v3/"}
	url := listAssignmentsURL(&client)
	if url != "http://localhost:5000/v3/role_assignments" {
		t.Errorf("Unexpected list URL generated: [%s]", url)
	}
}

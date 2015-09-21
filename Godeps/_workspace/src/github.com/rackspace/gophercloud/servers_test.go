package gophercloud

import (
	"net/http"
	"testing"
)

type testAccess struct {
	public, internal              string
	calledFirstEndpointByCriteria int
}

func (ta *testAccess) FirstEndpointUrlByCriteria(ac ApiCriteria) string {
	ta.calledFirstEndpointByCriteria++
	urls := []string{ta.public, ta.internal}
	return urls[ac.UrlChoice]
}

func (ta *testAccess) AuthToken() string {
	return ""
}

func (ta *testAccess) Revoke(string) error {
	return nil
}

func (ta *testAccess) Reauthenticate() error {
	return nil
}

func TestGetServersApi(t *testing.T) {
	c := TestContext().UseCustomClient(&http.Client{Transport: newTransport().WithResponse("Hello")})

	acc := &testAccess{
		public:   "http://localhost:8080",
		internal: "http://localhost:8086",
	}

	_, err := c.ServersApi(acc, ApiCriteria{
		Name:      "cloudComputeOpenStack",
		Region:    "dfw",
		VersionId: "2",
	})

	if err != nil {
		t.Error(err)
		return
	}

	if acc.calledFirstEndpointByCriteria != 1 {
		t.Error("Expected FirstEndpointByCriteria to be called")
		return
	}
}

package gophercloud

import (
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
)

func TestAuthenticatedHeaders(t *testing.T) {
	p := &ProviderClient{
		TokenID: "1234",
	}
	expected := map[string]string{"X-Auth-Token": "1234"}
	actual := p.AuthenticatedHeaders()
	th.CheckDeepEquals(t, expected, actual)
}

func TestUserAgent(t *testing.T) {
	p := &ProviderClient{}

	p.UserAgent.Prepend("custom-user-agent/v2.4")
	expected := "custom-user-agent/v2.4 gophercloud/v1.0"
	actual := p.UserAgent.Join()
	th.CheckEquals(t, expected, actual)

	p.UserAgent.Prepend("another-custom-user-agent/v0.3", "a-third-ua/v5.9")
	expected = "another-custom-user-agent/v0.3 a-third-ua/v5.9 custom-user-agent/v2.4 gophercloud/v1.0"
	actual = p.UserAgent.Join()
	th.CheckEquals(t, expected, actual)

	p.UserAgent = UserAgent{}
	expected = "gophercloud/v1.0"
	actual = p.UserAgent.Join()
	th.CheckEquals(t, expected, actual)
}

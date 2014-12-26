package gophercloud

import (
	"testing"

	th "github.com/rackspace/gophercloud/testhelper"
)

func TestServiceURL(t *testing.T) {
	c := &ServiceClient{Endpoint: "http://123.45.67.8/"}
	expected := "http://123.45.67.8/more/parts/here"
	actual := c.ServiceURL("more", "parts", "here")
	th.CheckEquals(t, expected, actual)
}

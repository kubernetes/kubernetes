package common

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/testhelper/client"
)

const TokenID = client.TokenID

func ServiceClient() *gophercloud.ServiceClient {
	return client.ServiceClient()
}

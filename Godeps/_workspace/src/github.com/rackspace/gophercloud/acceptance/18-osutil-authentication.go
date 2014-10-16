// +build acceptance,old

package main

import (
	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/osutil"
)

func main() {
	provider, authOptions, err := osutil.AuthOptions()
	if err != nil {
		panic(err)
	}
	_, err = gophercloud.Authenticate(provider, authOptions)
	if err != nil {
		panic(err)
	}
}

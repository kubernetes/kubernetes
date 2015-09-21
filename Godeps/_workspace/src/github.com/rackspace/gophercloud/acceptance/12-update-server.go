// +build acceptance,old

package main

import (
	"flag"
	"fmt"
	"github.com/rackspace/gophercloud"
)

var quiet = flag.Bool("quiet", false, "Quiet mode, for acceptance testing.  $? still indicates errors though.")

func main() {
	flag.Parse()
	withIdentity(false, func(acc gophercloud.AccessProvider) {
		withServerApi(acc, func(servers gophercloud.CloudServersProvider) {
			log("Creating server")
			id, err := createServer(servers, "", "", "", "")
			if err != nil {
				panic(err)
			}
			waitForServerState(servers, id, "ACTIVE")
			defer servers.DeleteServerById(id)

			log("Updating name of server")
			newName := randomString("ACPTTEST", 32)
			newDetails, err := servers.UpdateServer(id, gophercloud.NewServerSettings{
				Name: newName,
			})
			if err != nil {
				panic(err)
			}
			if newDetails.Name != newName {
				panic("Name change didn't appear to take")
			}

			log("Done")
		})
	})
}

func log(s string) {
	if !*quiet {
		fmt.Println(s)
	}
}

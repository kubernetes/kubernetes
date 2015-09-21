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

			log("Rescuing server")
			adminPass, err := servers.RescueServer(id)
			if err != nil {
				panic(err)
			}
			log("  Admin password = " + adminPass)
			if len(adminPass) < 1 {
				panic("Empty admin password")
			}
			waitForServerState(servers, id, "RESCUE")

			log("Unrescuing server")
			err = servers.UnrescueServer(id)
			if err != nil {
				panic(err)
			}
			waitForServerState(servers, id, "ACTIVE")

			log("Done")
		})
	})
}

func log(s string) {
	if !*quiet {
		fmt.Println(s)
	}
}

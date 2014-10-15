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
			serverId, err := createServer(servers, "", "", "", "")
			if err != nil {
				panic(err)
			}
			waitForServerState(servers, serverId, "ACTIVE")

			log("Soft-rebooting server")
			servers.RebootServer(serverId, false)
			waitForServerState(servers, serverId, "REBOOT")
			waitForServerState(servers, serverId, "ACTIVE")

			log("Hard-rebooting server")
			servers.RebootServer(serverId, true)
			waitForServerState(servers, serverId, "HARD_REBOOT")
			waitForServerState(servers, serverId, "ACTIVE")

			log("Done")
			servers.DeleteServerById(serverId)
		})
	})
}

func log(s string) {
	if !*quiet {
		fmt.Println(s)
	}
}

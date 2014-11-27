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

			log("Rebuilding server")
			newDetails, err := servers.RebuildServer(id, gophercloud.NewServer{
				Name:      randomString("ACPTTEST", 32),
				ImageRef:  findAlternativeImage(),
				FlavorRef: findAlternativeFlavor(),
				AdminPass: randomString("", 16),
			})
			if err != nil {
				panic(err)
			}
			waitForServerState(servers, newDetails.Id, "ACTIVE")

			log("Done")
		})
	})
}

func log(s string) {
	if !*quiet {
		fmt.Println(s)
	}
}

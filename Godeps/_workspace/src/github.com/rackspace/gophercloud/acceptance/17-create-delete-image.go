// +build acceptance,old

package main

import (
	"flag"
	"fmt"
	"github.com/rackspace/gophercloud"
)

var quiet = flag.Bool("quiet", false, "Quiet mode for acceptance testing.  $? non-zero on error though.")
var rgn = flag.String("r", "", "Datacenter region to interrogate.  Leave blank for provider-default region.")

func main() {
	flag.Parse()

	withIdentity(false, func(auth gophercloud.AccessProvider) {
		withServerApi(auth, func(servers gophercloud.CloudServersProvider) {
			log("Creating server")
			serverId, err := createServer(servers, "", "", "", "")
			if err != nil {
				panic(err)
			}
			waitForServerState(servers, serverId, "ACTIVE")

			log("Creating image")
			name := randomString("ACPTTEST", 16)
			createImage := gophercloud.CreateImage{
				Name: name,
			}
			imageId, err := servers.CreateImage(serverId, createImage)
			if err != nil {
				panic(err)
			}
			waitForImageState(servers, imageId, "ACTIVE")

			log("Deleting server")
			servers.DeleteServerById(serverId)

			log("Deleting image")
			servers.DeleteImageById(imageId)

			log("Done")
		})
	})
}

func log(s string) {
	if !*quiet {
		fmt.Println(s)
	}
}

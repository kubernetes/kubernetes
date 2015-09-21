// +build acceptance,old

package main

import (
	"flag"
	"fmt"
	"github.com/rackspace/gophercloud"
)

var region, serverName, imageRef, flavorRef *string
var adminPass = flag.String("a", "", "Administrator password (auto-assigned if none)")
var quiet = flag.Bool("quiet", false, "Quiet mode for acceptance tests.  $? non-zero if error.")

func configure() {
	region = flag.String("r", "", "Region in which to create the server.  Leave blank for provider-default region.")
	serverName = flag.String("n", randomString("ACPTTEST--", 16), "Server name (what you see in the control panel)")
	imageRef = flag.String("i", "", "ID of image to deploy onto the server")
	flavorRef = flag.String("f", "", "Flavor of server to deploy image upon")

	flag.Parse()
}

func main() {
	configure()

	withIdentity(false, func(auth gophercloud.AccessProvider) {
		withServerApi(auth, func(servers gophercloud.CloudServersProvider) {
			_, err := createServer(servers, *imageRef, *flavorRef, *serverName, *adminPass)
			if err != nil {
				panic(err)
			}

			allServers, err := servers.ListServers()
			if err != nil {
				panic(err)
			}

			if !*quiet {
				fmt.Printf("ID,Name,Status,Progress\n")
				for _, i := range allServers {
					fmt.Printf("%s,\"%s\",%s,%d\n", i.Id, i.Name, i.Status, i.Progress)
				}
			}
		})
	})
}

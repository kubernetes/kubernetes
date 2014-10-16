// +build acceptance,old

package main

import (
	"flag"
	"fmt"
	"github.com/rackspace/gophercloud"
)

var quiet = flag.Bool("quiet", false, "Quiet mode, for acceptance testing.  $? still indicates errors though.")
var serverId = flag.String("i", "", "ID of server whose admin password is to be changed.")
var newPass = flag.String("p", "", "New password for the server.")

func main() {
	flag.Parse()

	withIdentity(false, func(acc gophercloud.AccessProvider) {
		withServerApi(acc, func(api gophercloud.CloudServersProvider) {
			// If user doesn't explicitly provide a server ID, create one dynamically.
			if *serverId == "" {
				var err error
				*serverId, err = createServer(api, "", "", "", "")
				if err != nil {
					panic(err)
				}
				waitForServerState(api, *serverId, "ACTIVE")
			}

			// If no password is provided, create one dynamically.
			if *newPass == "" {
				*newPass = randomString("", 16)
			}

			// Submit the request for changing the admin password.
			// Note that we don't verify this actually completes;
			// doing so is beyond the scope of the SDK, and should be
			// the responsibility of your specific OpenStack provider.
			err := api.SetAdminPassword(*serverId, *newPass)
			if err != nil {
				panic(err)
			}

			if !*quiet {
				fmt.Println("Password change request submitted.")
			}
		})
	})
}

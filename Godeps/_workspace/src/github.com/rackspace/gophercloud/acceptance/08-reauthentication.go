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

	// Invoke withIdentity such that re-auth is enabled.
	withIdentity(true, func(auth gophercloud.AccessProvider) {
		token1 := auth.AuthToken()

		withServerApi(auth, func(servers gophercloud.CloudServersProvider) {
			// Just to confirm everything works, we should be able to list images without error.
			_, err := servers.ListImages()
			if err != nil {
				panic(err)
			}

			// Revoke our current authentication token.
			auth.Revoke(auth.AuthToken())

			// Attempt to list images again.  This should _succeed_, because we enabled re-authentication.
			_, err = servers.ListImages()
			if err != nil {
				panic(err)
			}

			// However, our new authentication token should differ.
			token2 := auth.AuthToken()

			if !*quiet {
				fmt.Println("Old authentication token: ", token1)
				fmt.Println("New authentication token: ", token2)
			}

			if token1 == token2 {
				panic("Tokens should differ")
			}
		})
	})
}

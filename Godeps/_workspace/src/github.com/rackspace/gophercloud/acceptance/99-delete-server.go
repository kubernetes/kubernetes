// +build acceptance,old

package main

import (
	"flag"
	"fmt"
	"github.com/rackspace/gophercloud"
)

var quiet = flag.Bool("quiet", false, "Quiet operation for acceptance tests.  $? non-zero if problem.")
var region = flag.String("r", "", "Datacenter region.  Leave blank for provider-default region.")

func main() {
	flag.Parse()

	withIdentity(false, func(auth gophercloud.AccessProvider) {
		withServerApi(auth, func(servers gophercloud.CloudServersProvider) {
			// Grab a listing of all servers.
			ss, err := servers.ListServers()
			if err != nil {
				panic(err)
			}

			// And for each one that starts with the ACPTTEST prefix, delete it.
			// These are likely left-overs from previously running acceptance tests.
			// Note that 04-create-servers.go is intended to leak servers by intention,
			// so as to test this code.  :)
			n := 0
			for _, s := range ss {
				if len(s.Name) < 8 {
					continue
				}
				if s.Name[0:8] == "ACPTTEST" {
					err := servers.DeleteServerById(s.Id)
					if err != nil {
						panic(err)
					}
					n++
				}
			}

			if !*quiet {
				fmt.Printf("%d servers removed.\n", n)
			}
		})
	})
}

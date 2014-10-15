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
			keypairs, err := servers.ListKeyPairs()
			if err != nil {
				panic(err)
			}

			if !*quiet {
				fmt.Println("name,fingerprint,publickey")
				for _, key := range keypairs {
					fmt.Printf("%s,%s,%s\n", key.Name, key.FingerPrint, key.PublicKey)
				}
			}
		})
	})
}

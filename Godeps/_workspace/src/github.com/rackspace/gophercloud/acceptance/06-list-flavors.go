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
			flavors, err := servers.ListFlavors()
			if err != nil {
				panic(err)
			}

			if !*quiet {
				fmt.Println("ID,Name,MinRam,MinDisk")
				for _, f := range flavors {
					fmt.Printf("%s,\"%s\",%d,%d\n", f.Id, f.Name, f.Ram, f.Disk)
				}
			}
		})
	})
}

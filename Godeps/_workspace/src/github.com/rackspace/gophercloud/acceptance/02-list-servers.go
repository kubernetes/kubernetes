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
		withServerApi(acc, func(api gophercloud.CloudServersProvider) {
			tryFullDetails(api)
			tryLinksOnly(api)
		})
	})
}

func tryLinksOnly(api gophercloud.CloudServersProvider) {
	servers, err := api.ListServersLinksOnly()
	if err != nil {
		panic(err)
	}

	if !*quiet {
		fmt.Println("Id,Name")
		for _, s := range servers {
			if s.AccessIPv4 != "" {
				panic("IPv4 not expected")
			}

			if s.Status != "" {
				panic("Status not expected")
			}

			if s.Progress != 0 {
				panic("Progress not expected")
			}

			fmt.Printf("%s,\"%s\"\n", s.Id, s.Name)
		}
	}
}

func tryFullDetails(api gophercloud.CloudServersProvider) {
	servers, err := api.ListServers()
	if err != nil {
		panic(err)
	}

	if !*quiet {
		fmt.Println("Id,Name,AccessIPv4,Status,Progress")
		for _, s := range servers {
			fmt.Printf("%s,\"%s\",%s,%s,%d\n", s.Id, s.Name, s.AccessIPv4, s.Status, s.Progress)
		}
	}
}

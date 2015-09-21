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
			log("Creating server")
			id, err := createServer(api, "", "", "", "")
			if err != nil {
				panic(err)
			}
			waitForServerState(api, id, "ACTIVE")
			defer api.DeleteServerById(id)

			tryAllAddresses(id, api)

			log("Done")
		})
	})
}

func tryAllAddresses(id string, api gophercloud.CloudServersProvider) {
	log("Getting the server instance")
	s, err := api.ServerById(id)
	if err != nil {
		panic(err)
	}

	log("Getting the complete set of pools")
	ps, err := s.AllAddressPools()
	if err != nil {
		panic(err)
	}

	log("Listing IPs for each pool")
	for k, v := range ps {
		log(fmt.Sprintf("  Pool %s", k))
		for _, a := range v {
			log(fmt.Sprintf("    IP: %s, Version: %d", a.Addr, a.Version))
		}
	}
}

func log(s ...interface{}) {
	if !*quiet {
		fmt.Println(s...)
	}
}

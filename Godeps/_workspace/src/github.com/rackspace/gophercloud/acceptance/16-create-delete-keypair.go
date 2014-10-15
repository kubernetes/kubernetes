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
			name := randomString("ACPTTEST", 16)
			kp := gophercloud.NewKeyPair{
				Name: name,
			}
			keypair, err := servers.CreateKeyPair(kp)
			if err != nil {
				panic(err)
			}
			if !*quiet {
				fmt.Printf("%s,%s,%s\n", keypair.Name, keypair.FingerPrint, keypair.PublicKey)
			}

			keypair, err = servers.ShowKeyPair(name)
			if err != nil {
				panic(err)
			}
			if !*quiet {
				fmt.Printf("%s,%s,%s\n", keypair.Name, keypair.FingerPrint, keypair.PublicKey)
			}

			err = servers.DeleteKeyPair(name)
			if err != nil {
				panic(err)
			}
		})
	})
}

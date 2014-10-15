// +build acceptance,old

package main

import (
	"flag"
	"fmt"
	"github.com/rackspace/gophercloud"
	"time"
)

var quiet = flag.Bool("quiet", false, "Quiet mode, for acceptance testing.  $? still indicates errors though.")

func main() {
	flag.Parse()

	withIdentity(false, func(acc gophercloud.AccessProvider) {
		withServerApi(acc, func(api gophercloud.CloudServersProvider) {
			// These tests are going to take some time to complete.
			// So, we'll do two tests at the same time to help amortize test time.
			done := make(chan bool)
			go resizeRejectTest(api, done)
			go resizeAcceptTest(api, done)
			_ = <-done
			_ = <-done

			if !*quiet {
				fmt.Println("Done.")
			}
		})
	})
}

// Perform the resize test, but reject the resize request.
func resizeRejectTest(api gophercloud.CloudServersProvider, done chan bool) {
	withServer(api, func(id string) {
		newFlavorId := findAlternativeFlavor()
		err := api.ResizeServer(id, randomString("ACPTTEST", 24), newFlavorId, "")
		if err != nil {
			panic(err)
		}

		waitForServerState(api, id, "VERIFY_RESIZE")

		err = api.RevertResize(id)
		if err != nil {
			panic(err)
		}
	})
	done <- true
}

// Perform the resize test, but accept the resize request.
func resizeAcceptTest(api gophercloud.CloudServersProvider, done chan bool) {
	withServer(api, func(id string) {
		newFlavorId := findAlternativeFlavor()
		err := api.ResizeServer(id, randomString("ACPTTEST", 24), newFlavorId, "")
		if err != nil {
			panic(err)
		}

		waitForServerState(api, id, "VERIFY_RESIZE")

		err = api.ConfirmResize(id)
		if err != nil {
			panic(err)
		}
	})
	done <- true
}

func withServer(api gophercloud.CloudServersProvider, f func(string)) {
	id, err := createServer(api, "", "", "", "")
	if err != nil {
		panic(err)
	}

	for {
		s, err := api.ServerById(id)
		if err != nil {
			panic(err)
		}
		if s.Status == "ACTIVE" {
			break
		}
		time.Sleep(10 * time.Second)
	}

	f(id)

	// I've learned that resizing an instance can fail if a delete request
	// comes in prior to its completion.  This ends up leaving the server
	// in an error state, and neither the resize NOR the delete complete.
	// This is a bug in OpenStack, as far as I'm concerned, but thankfully,
	// there's an easy work-around -- just wait for your server to return to
	// active state first!
	waitForServerState(api, id, "ACTIVE")
	err = api.DeleteServerById(id)
	if err != nil {
		panic(err)
	}
}

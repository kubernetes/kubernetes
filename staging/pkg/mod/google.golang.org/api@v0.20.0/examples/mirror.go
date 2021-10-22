// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	mirror "google.golang.org/api/mirror/v1"
)

const mirrorLayout = "Jan 2, 2006 at 3:04pm"

func init() {
	scopes := []string{
		mirror.GlassLocationScope,
		mirror.GlassTimelineScope,
	}
	registerDemo("mirror", strings.Join(scopes, " "), mirrorMain)
}

// mirrorMain is an example that demonstrates calling the Mirror API.
//
// Example usage:
//   go build -o go-api-demo *.go
//   go-api-demo -clientid="my-clientid" -secret="my-secret" mirror
func mirrorMain(client *http.Client, argv []string) {
	if len(argv) != 0 {
		fmt.Fprintln(os.Stderr, "Usage: mirror")
		return
	}

	svc, err := mirror.New(client)
	if err != nil {
		log.Fatalf("Unable to create Mirror service: %v", err)
	}

	cs, err := svc.Contacts.List().Do()
	if err != nil {
		log.Fatalf("Unable to retrieve glass contacts: %v", err)
	}

	if len(cs.Items) == 0 {
		log.Printf("You have no glass contacts.  Let's add one.")
		mom := &mirror.Contact{
			DisplayName:   "Mom",
			Id:            "mom",
			PhoneNumber:   "123-456-7890",
			SpeakableName: "mom",
		}
		_, err := svc.Contacts.Insert(mom).Do()
		if err != nil {
			log.Fatalf("Unable to add %v to glass contacts: %v", mom.DisplayName, err)
		}
	}
	for _, c := range cs.Items {
		log.Printf("Found glass contact %q, phone number: %v", c.DisplayName, c.PhoneNumber)
	}

	ls, err := svc.Locations.List().Do()
	if err != nil {
		log.Fatalf("Unable to retrieve glass locations: %v", err)
	}

	if len(ls.Items) == 0 {
		log.Printf("You have no glass locations.")
	}
	for _, loc := range ls.Items {
		t, err := time.Parse(time.RFC3339, loc.Timestamp)
		if err != nil {
			log.Printf("unable to parse time %q: %v", loc.Timestamp, err)
		}
		log.Printf("Found glass location: %q at %v, address: %v (lat=%v, lon=%v)", loc.DisplayName, t.Format(mirrorLayout), loc.Address, loc.Latitude, loc.Longitude)
	}

	ts, err := svc.Timeline.List().Do()
	if err != nil {
		log.Fatalf("Unable to retrieve glass timeline: %v", err)
	}

	if len(ts.Items) == 0 {
		log.Printf("You have no glass timeline items.")
	}
	for _, v := range ts.Items {
		t, err := time.Parse(time.RFC3339, v.Updated)
		if err != nil {
			log.Printf("unable to parse time %q: %v", v.Updated, err)
		}
		log.Printf("Found glass timeline item: %q at %v", v.Text, t.Format(mirrorLayout))
	}
}

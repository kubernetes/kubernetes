// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"log"
	"net/http"
	"os"

	calendar "google.golang.org/api/calendar/v3"
)

func init() {
	registerDemo("calendar", calendar.CalendarScope, calendarMain)
}

// calendarMain is an example that demonstrates calling the Calendar API.
// Its purpose is to test out the ability to get maps of struct objects.
//
// Example usage:
//   go build -o go-api-demo *.go
//   go-api-demo -clientid="my-clientid" -secret="my-secret" calendar
func calendarMain(client *http.Client, argv []string) {
	if len(argv) != 0 {
		fmt.Fprintln(os.Stderr, "Usage: calendar")
		return
	}

	svc, err := calendar.New(client)
	if err != nil {
		log.Fatalf("Unable to create Calendar service: %v", err)
	}

	c, err := svc.Colors.Get().Do()
	if err != nil {
		log.Fatalf("Unable to retrieve calendar colors: %v", err)
	}

	log.Printf("Kind of colors: %v", c.Kind)
	log.Printf("Colors last updated: %v", c.Updated)

	for k, v := range c.Calendar {
		log.Printf("Calendar[%v]: Background=%v, Foreground=%v", k, v.Background, v.Foreground)
	}

	for k, v := range c.Event {
		log.Printf("Event[%v]: Background=%v, Foreground=%v", k, v.Background, v.Foreground)
	}

	listRes, err := svc.CalendarList.List().Fields("items/id").Do()
	if err != nil {
		log.Fatalf("Unable to retrieve list of calendars: %v", err)
	}
	for _, v := range listRes.Items {
		log.Printf("Calendar ID: %v\n", v.Id)
	}

	if len(listRes.Items) > 0 {
		id := listRes.Items[0].Id
		res, err := svc.Events.List(id).Fields("items(updated,summary)", "summary", "nextPageToken").Do()
		if err != nil {
			log.Fatalf("Unable to retrieve calendar events list: %v", err)
		}
		for _, v := range res.Items {
			log.Printf("Calendar ID %q event: %v: %q\n", id, v.Updated, v.Summary)
		}
		log.Printf("Calendar ID %q Summary: %v\n", id, res.Summary)
		log.Printf("Calendar ID %q next page token: %v\n", id, res.NextPageToken)
	}
}

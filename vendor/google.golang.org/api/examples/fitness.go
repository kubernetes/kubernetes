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

	fitness "google.golang.org/api/fitness/v1"
)

const (
	layout        = "Jan 2, 2006 at 3:04pm" // for time.Format
	nanosPerMilli = 1e6
)

func init() {
	scopes := []string{
		fitness.FitnessActivityReadScope,
		fitness.FitnessActivityWriteScope,
		fitness.FitnessBodyReadScope,
		fitness.FitnessBodyWriteScope,
		fitness.FitnessLocationReadScope,
		fitness.FitnessLocationWriteScope,
	}
	registerDemo("fitness", strings.Join(scopes, " "), fitnessMain)
}

// millisToTime converts Unix millis to time.Time.
func millisToTime(t int64) time.Time {
	return time.Unix(0, t*nanosPerMilli)
}

// fitnessMain is an example that demonstrates calling the Fitness API.
//
// Example usage:
//   go build -o go-api-demo *.go
//   go-api-demo -clientid="my-clientid" -secret="my-secret" fitness
func fitnessMain(client *http.Client, argv []string) {
	if len(argv) != 0 {
		fmt.Fprintln(os.Stderr, "Usage: fitness")
		return
	}

	svc, err := fitness.New(client)
	if err != nil {
		log.Fatalf("Unable to create Fitness service: %v", err)
	}

	us, err := svc.Users.Sessions.List("me").Do()
	if err != nil {
		log.Fatalf("Unable to retrieve user's sessions: %v", err)
	}
	if len(us.Session) == 0 {
		log.Fatal("You have no user sessions to explore.")
	}

	var minTime, maxTime time.Time
	for _, s := range us.Session {
		start := millisToTime(s.StartTimeMillis)
		end := millisToTime(s.EndTimeMillis)
		if minTime.IsZero() || start.Before(minTime) {
			minTime = start
		}
		if maxTime.IsZero() || end.After(maxTime) {
			maxTime = end
		}
		log.Printf("Session %q, %v - %v, activity type=%v", s.Name, start.Format(layout), end.Format(layout), s.ActivityType)
	}

	ds, err := svc.Users.DataSources.List("me").Do()
	if err != nil {
		log.Fatalf("Unable to retrieve user's data sources: %v", err)
	}
	if len(ds.DataSource) == 0 {
		log.Fatal("You have no data sources to explore.")
	}
	for _, d := range ds.DataSource {
		format := "integer"
		if d.DataType != nil && len(d.DataType.Field) > 0 {
			f := d.DataType.Field[0]
			format = f.Format
			log.Printf("Data source %q, name %q is of type %q", d.DataStreamName, f.Name, format)
		} else {
			log.Printf("Data source %q is of type %q", d.DataStreamName, d.Type)
		}
		setID := fmt.Sprintf("%v-%v", minTime.UnixNano(), maxTime.UnixNano())
		data, err := svc.Users.DataSources.Datasets.Get("me", d.DataStreamId, setID).Do()
		if err != nil {
			log.Fatalf("Unable to retrieve user's data source stream %v, %v: %v", d.DataStreamId, setID, err)
		}
		for _, p := range data.Point {
			for _, v := range p.Value {
				t := millisToTime(p.ModifiedTimeMillis).Format(layout)
				if format == "integer" {
					log.Printf("data at %v = %v", t, v.IntVal)
				} else {
					log.Printf("data at %v = %v", t, v.FpVal)
				}
			}
		}
	}
}

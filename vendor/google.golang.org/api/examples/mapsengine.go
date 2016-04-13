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

	mapsengine "google.golang.org/api/mapsengine/v1"
)

func init() {
	scopes := []string{
		mapsengine.MapsengineScope,
		mapsengine.MapsengineReadonlyScope,
	}
	registerDemo("mapsengine", strings.Join(scopes, " "), mapsengineMain)
}

func showMapFeatures(svc *mapsengine.Service, id string) {
	r, err := svc.Tables.Get(id).Version("published").Do()
	if err != nil {
		log.Fatalf("Unable to get map %v table: %v", id, err)
	}
	fmt.Printf("Map ID: %v, Name: %q, Description: %q\n", id, r.Name, r.Description)

	pageToken := ""
	for {
		time.Sleep(1 * time.Second) // Don't violate free rate limit
		// Read the location of every Feature in a Table.
		req := svc.Tables.Features.List(id).MaxResults(500).Version("published")
		if pageToken != "" {
			req.PageToken(pageToken)
		}
		r, err := req.Do()
		if err != nil {
			log.Fatalf("Unable to list table features: %v", err)
		}

		for _, f := range r.Features {
			if v, ok := f.Geometry.GeometryCollection(); ok {
				fmt.Printf("%v: %v\n", f.Geometry.Type(), v)
			} else if v, ok := f.Geometry.LineString(); ok {
				fmt.Printf("%v: %v\n", f.Geometry.Type(), v)
			} else if v, ok := f.Geometry.MultiLineString(); ok {
				fmt.Printf("%v: %v\n", f.Geometry.Type(), v)
			} else if v, ok := f.Geometry.MultiPoint(); ok {
				fmt.Printf("%v: %v\n", f.Geometry.Type(), v)
			} else if v, ok := f.Geometry.MultiPolygon(); ok {
				fmt.Printf("%v: %v\n", f.Geometry.Type(), v)
			} else if v, ok := f.Geometry.Point(); ok {
				fmt.Printf("%v: %v\n", f.Geometry.Type(), v)
			} else if v, ok := f.Geometry.Polygon(); ok {
				fmt.Printf("%v: %v\n", f.Geometry.Type(), v)
			} else {
				log.Fatalf("Unknown GeoJsonGeometry type %q", f.Geometry.Type())
			}
		}

		if r.NextPageToken == "" {
			break
		}
		pageToken = r.NextPageToken
	}
}

// mapsengineMain is an example that demonstrates calling the Mapsengine API.
// Please see https://developers.google.com/maps-engine/documentation/hello-world#go
// for more information.
//
// Example usage:
//   go build -o go-api-demo *.go
//   go-api-demo -clientid="my-clientid" -secret="my-secret" mapsengine
func mapsengineMain(client *http.Client, argv []string) {
	if len(argv) != 0 {
		fmt.Fprintln(os.Stderr, "Usage: mapsengine")
		return
	}

	svc, err := mapsengine.New(client)
	if err != nil {
		log.Fatalf("Unable to create Mapsengine service: %v", err)
	}

	showMapFeatures(svc, "14137585153106784136-16071188762309719429")
	showMapFeatures(svc, "12421761926155747447-06672618218968397709")
}

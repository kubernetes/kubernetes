/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// A tiny web server for viewing the environment kubernetes creates for your
// containers. It exposes the filesystem and environment variables via http
// server.
package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"

	"github.com/davecgh/go-spew/spew"
)

var (
	port = flag.Int("port", 8080, "Port number to serve at.")
)

func main() {
	flag.Parse()
	hostname, err := os.Hostname()
	if err != nil {
		log.Fatalf("Error getting hostname: %v", err)
	}

	links := []struct {
		link, desc string
	}{
		{"/fs/", "Complete file system as seen by this container."},
		{"/vars/", "Environment variables as seen by this container."},
		{"/hostname/", "Hostname as seen by this container."},
		{"/dns?q=google.com", "Explore DNS records seen by this container."},
		{"/quit", "Cause this container to exit."},
	}

	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, "<b> Kubernetes environment explorer </b><br/><br/>")
		for _, v := range links {
			fmt.Fprintf(w, `<a href="%v">%v: %v</a><br/>`, v.link, v.link, v.desc)
		}
	})

	http.Handle("/fs/", http.StripPrefix("/fs/", http.FileServer(http.Dir("/"))))
	http.HandleFunc("/vars/", func(w http.ResponseWriter, r *http.Request) {
		for _, v := range os.Environ() {
			fmt.Fprintf(w, "%v\n", v)
		}
	})
	http.HandleFunc("/hostname/", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintf(w, hostname)
	})
	http.HandleFunc("/quit", func(w http.ResponseWriter, r *http.Request) {
		os.Exit(0)
	})
	http.HandleFunc("/dns", dns)

	go log.Fatal(http.ListenAndServe(fmt.Sprintf("0.0.0.0:%d", *port), nil))

	select {}
}

func dns(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query().Get("q")
	// Note that the below is NOT safe from input attacks, but that's OK
	// because this is just for debugging.
	fmt.Fprintf(w, `<html><body>
<form action="/dns">
<input name="q" type="text" value="%v"></input>
<button type="submit">Lookup</button>
</form>
<br/><br/><pre>`, q)
	{
		res, err := net.LookupNS(q)
		spew.Fprintf(w, "LookupNS(%v):\nResult: %#v\nError: %v\n\n", q, res, err)
	}
	{
		res, err := net.LookupTXT(q)
		spew.Fprintf(w, "LookupTXT(%v):\nResult: %#v\nError: %v\n\n", q, res, err)
	}
	{
		cname, res, err := net.LookupSRV("", "", q)
		spew.Fprintf(w, `LookupSRV("", "", %v):
cname: %v
Result: %#v
Error: %v

`, q, cname, res, err)
	}
	{
		res, err := net.LookupHost(q)
		spew.Fprintf(w, "LookupHost(%v):\nResult: %#v\nError: %v\n\n", q, res, err)
	}
	{
		res, err := net.LookupIP(q)
		spew.Fprintf(w, "LookupIP(%v):\nResult: %#v\nError: %v\n\n", q, res, err)
	}
	{
		res, err := net.LookupMX(q)
		spew.Fprintf(w, "LookupMX(%v):\nResult: %#v\nError: %v\n\n", q, res, err)
	}
	fmt.Fprintf(w, `</pre>
</body>
</html>`)
}

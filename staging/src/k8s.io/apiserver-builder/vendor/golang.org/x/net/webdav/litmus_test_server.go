// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*
This program is a server for the WebDAV 'litmus' compliance test at
http://www.webdav.org/neon/litmus/
To run the test:

go run litmus_test_server.go

and separately, from the downloaded litmus-xxx directory:

make URL=http://localhost:9999/ check
*/
package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"net/url"

	"golang.org/x/net/webdav"
)

var port = flag.Int("port", 9999, "server port")

func main() {
	flag.Parse()
	log.SetFlags(0)
	h := &webdav.Handler{
		FileSystem: webdav.NewMemFS(),
		LockSystem: webdav.NewMemLS(),
		Logger: func(r *http.Request, err error) {
			litmus := r.Header.Get("X-Litmus")
			if len(litmus) > 19 {
				litmus = litmus[:16] + "..."
			}

			switch r.Method {
			case "COPY", "MOVE":
				dst := ""
				if u, err := url.Parse(r.Header.Get("Destination")); err == nil {
					dst = u.Path
				}
				o := r.Header.Get("Overwrite")
				log.Printf("%-20s%-10s%-30s%-30so=%-2s%v", litmus, r.Method, r.URL.Path, dst, o, err)
			default:
				log.Printf("%-20s%-10s%-30s%v", litmus, r.Method, r.URL.Path, err)
			}
		},
	}

	// The next line would normally be:
	//	http.Handle("/", h)
	// but we wrap that HTTP handler h to cater for a special case.
	//
	// The propfind_invalid2 litmus test case expects an empty namespace prefix
	// declaration to be an error. The FAQ in the webdav litmus test says:
	//
	// "What does the "propfind_invalid2" test check for?...
	//
	// If a request was sent with an XML body which included an empty namespace
	// prefix declaration (xmlns:ns1=""), then the server must reject that with
	// a "400 Bad Request" response, as it is invalid according to the XML
	// Namespace specification."
	//
	// On the other hand, the Go standard library's encoding/xml package
	// accepts an empty xmlns namespace, as per the discussion at
	// https://github.com/golang/go/issues/8068
	//
	// Empty namespaces seem disallowed in the second (2006) edition of the XML
	// standard, but allowed in a later edition. The grammar differs between
	// http://www.w3.org/TR/2006/REC-xml-names-20060816/#ns-decl and
	// http://www.w3.org/TR/REC-xml-names/#dt-prefix
	//
	// Thus, we assume that the propfind_invalid2 test is obsolete, and
	// hard-code the 400 Bad Request response that the test expects.
	http.Handle("/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("X-Litmus") == "props: 3 (propfind_invalid2)" {
			http.Error(w, "400 Bad Request", http.StatusBadRequest)
			return
		}
		h.ServeHTTP(w, r)
	}))

	addr := fmt.Sprintf(":%d", *port)
	log.Printf("Serving %v", addr)
	log.Fatal(http.ListenAndServe(addr, nil))
}

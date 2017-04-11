// Copyright 2016 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*
cbtemulator launches the in-memory Cloud Bigtable server on the given address.
*/
package main

import (
	"flag"
	"fmt"
	"log"

	"cloud.google.com/go/bigtable/bttest"
)

var (
	host = flag.String("host", "localhost", "the address to bind to on the local machine")
	port = flag.Int("port", 9000, "the port number to bind to on the local machine")
)

func main() {
	flag.Parse()
	srv, err := bttest.NewServer(fmt.Sprintf("%s:%d", *host, *port))
	if err != nil {
		log.Fatalf("failed to start emulator: %v", err)
	}

	fmt.Printf("Cloud Bigtable emulator running on %s\n", srv.Addr)
	select {}
}

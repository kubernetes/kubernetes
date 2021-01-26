/*
Copyright 2020 The Kubernetes Authors.

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

// package main connects to the given host and port and redirects its stdin to the connection and
// the connection's output to stdout. This is currently being used for port-forwarding for Windows Pods.
package main

import (
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"sync"
)

func main() {
	if len(os.Args) != 3 {
		log.Fatalln("usage: wincat <host> <port>")
	}
	host := os.Args[1]
	port := os.Args[2]

	addr, err := net.ResolveTCPAddr("tcp", fmt.Sprintf("%s:%s", host, port))
	if err != nil {
		log.Fatalf("Failed to resolve TCP addr %v %v", host, port)
	}

	conn, err := net.DialTCP("tcp", nil, addr)
	if err != nil {
		log.Fatalf("Failed to connect to %s:%s because %s", host, port, err)
	}
	defer func() {
		if err := conn.Close(); err != nil {
			log.Printf("error while closing conn stream: %v", err)
		}
	}()

	var wg sync.WaitGroup
	wg.Add(2)

	go func() {
		defer func() {
			if err := os.Stdout.Close(); err != nil {
				log.Printf("error while closing stdout stream: %v", err)
			}
			if err := os.Stdin.Close(); err != nil {
				log.Printf("error while closing stdin stream: %v", err)
			}
			if err := conn.CloseRead(); err != nil {
				log.Printf("error while closing conn read stream: %v", err)
			}
			wg.Done()
		}()

		_, err := io.Copy(os.Stdout, conn)
		if err != nil {
			log.Printf("error while copying stream to stdout: %v", err)
		}
	}()

	go func() {
		defer func() {
			if err := conn.CloseWrite(); err != nil {
				log.Printf("error while closing conn write stream: %v", err)
			}
			wg.Done()
		}()

		_, err := io.Copy(conn, os.Stdin)
		if err != nil {
			log.Printf("error while copying stream from stdin: %v", err)
		}
	}()
	wg.Wait()
}

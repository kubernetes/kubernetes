/*
Copyright 2014 The Kubernetes Authors.

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

// A small utility to just serve the hostname on TCP and/or UDP.
package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

var (
	doTCP  = flag.Bool("tcp", false, "Serve raw over TCP.")
	doUDP  = flag.Bool("udp", false, "Serve raw over UDP.")
	doHTTP = flag.Bool("http", true, "Serve HTTP.")
	port   = flag.Int("port", 9376, "Port number.")
)

func main() {
	flag.Parse()
	if *doHTTP && (*doTCP || *doUDP) {
		log.Fatalf("Can't server TCP/UDP mode and HTTP mode at the same time")
	}

	hostname, err := os.Hostname()
	if err != nil {
		log.Fatalf("Error from os.Hostname(): %s", err)
	}

	if *doTCP {
		listener, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
		if err != nil {
			log.Fatalf("Error from net.Listen(): %s", err)
		}
		go func() {
			for {
				conn, err := listener.Accept()
				if err != nil {
					log.Fatalf("Error from Accept(): %s", err)
				}
				log.Printf("TCP request from %s", conn.RemoteAddr().String())
				conn.Write([]byte(hostname))
				conn.Close()
			}
		}()
	}
	if *doUDP {
		addr, err := net.ResolveUDPAddr("udp", fmt.Sprintf(":%d", *port))
		if err != nil {
			log.Fatalf("Error from net.ResolveUDPAddr(): %s", err)
		}
		sock, err := net.ListenUDP("udp", addr)
		if err != nil {
			log.Fatalf("Error from ListenUDP(): %s", err)
		}
		go func() {
			var buffer [16]byte
			for {
				_, cliAddr, err := sock.ReadFrom(buffer[0:])
				if err != nil {
					log.Fatalf("Error from ReadFrom(): %s", err)
				}
				log.Printf("UDP request from %s", cliAddr.String())
				sock.WriteTo([]byte(hostname), cliAddr)
			}
		}()
	}
	if *doHTTP {
		http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
			log.Printf("HTTP request from %s", r.RemoteAddr)
			fmt.Fprintf(w, "%s", hostname)
		})
		go func() {
			// Run in a closure so http.ListenAndServe doesn't block
			log.Fatal(http.ListenAndServe(fmt.Sprintf(":%d", *port), nil))
		}()
	}
	log.Printf("Serving on port %d.\n", *port)
	signals := make(chan os.Signal)
	signal.Notify(signals, syscall.SIGTERM)
	sig := <-signals
	log.Printf("Shutting down after receiving signal: %s.\n", sig)
	log.Printf("Awaiting pod deletion.\n")
	time.Sleep(60 * time.Second)
}

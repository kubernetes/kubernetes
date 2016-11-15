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

// A tiny binary for testing port forwarding. The following environment variables
// control the binary's logic:
//
// BIND_PORT - the TCP port to use for the listener
// EXPECTED_CLIENT_DATA - data that we expect to receive from the client; may be "".
// CHUNKS - how many chunks of data we should send to the client
// CHUNK_SIZE - how large each chunk should be
// CHUNK_INTERVAL - the delay in between sending each chunk
//
// Log messages are written to stdout at various stages of the binary's execution.
// Test code can retrieve this container's log and validate that the expected
// behavior is taking place.
package main

import (
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"time"
)

func getEnvInt(name string) int {
	s := os.Getenv(name)
	value, err := strconv.Atoi(s)
	if err != nil {
		fmt.Printf("Error parsing %s %q: %v\n", name, s, err)
		os.Exit(1)
	}
	return value
}

func main() {
	bindAddress := os.Getenv("BIND_ADDRESS")
	if bindAddress == "" {
		bindAddress = "localhost"
	}
	bindPort := os.Getenv("BIND_PORT")
	listener, err := net.Listen("tcp", fmt.Sprintf("%s:%s", bindAddress, bindPort))
	if err != nil {
		fmt.Printf("Error listening: %v\n", err)
		os.Exit(1)
	}

	conn, err := listener.Accept()
	if err != nil {
		fmt.Printf("Error accepting connection: %v\n", err)
		os.Exit(1)
	}
	defer conn.Close()
	fmt.Println("Accepted client connection")

	expectedClientData := os.Getenv("EXPECTED_CLIENT_DATA")
	if len(expectedClientData) > 0 {
		buf := make([]byte, len(expectedClientData))
		read, err := conn.Read(buf)
		if read != len(expectedClientData) {
			fmt.Printf("Expected to read %d bytes from client, but got %d instead. err=%v\n", len(expectedClientData), read, err)
			os.Exit(2)
		}
		if expectedClientData != string(buf) {
			fmt.Printf("Expect to read %q, but got %q. err=%v\n", expectedClientData, string(buf), err)
			os.Exit(3)
		}
		if err != nil {
			fmt.Printf("Read err: %v\n", err)
		}
		fmt.Println("Received expected client data")
	}

	chunks := getEnvInt("CHUNKS")
	chunkSize := getEnvInt("CHUNK_SIZE")
	chunkInterval := getEnvInt("CHUNK_INTERVAL")

	stringData := strings.Repeat("x", chunkSize)
	data := []byte(stringData)

	for i := 0; i < chunks; i++ {
		written, err := conn.Write(data)
		if written != chunkSize {
			fmt.Printf("Expected to write %d bytes from client, but wrote %d instead. err=%v\n", chunkSize, written, err)
			os.Exit(4)
		}
		if err != nil {
			fmt.Printf("Write err: %v\n", err)
		}
		if i+1 < chunks {
			time.Sleep(time.Duration(chunkInterval) * time.Millisecond)
		}
	}

	fmt.Println("Done")
}

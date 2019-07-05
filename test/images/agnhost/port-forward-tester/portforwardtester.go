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

package portforwardtester

import (
	"fmt"
	"net"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/spf13/cobra"
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

// taken from net/http/server.go:
//
// rstAvoidanceDelay is the amount of time we sleep after closing the
// write side of a TCP connection before closing the entire socket.
// By sleeping, we increase the chances that the client sees our FIN
// and processes its final data before they process the subsequent RST
// from closing a connection with known unread data.
// This RST seems to occur mostly on BSD systems. (And Windows?)
// This timeout is somewhat arbitrary (~latency around the planet).
const rstAvoidanceDelay = 500 * time.Millisecond

// CmdPortForwardTester is used by agnhost Cobra.
var CmdPortForwardTester = &cobra.Command{
	Use:   "port-forward-tester",
	Short: "Creates a TCP server that sends chunks of data",
	Long: `Listens for TCP connections on a given address and port, optionally checks the data received,
and sends a configurable number of data chunks, with a configurable interval between chunks.

The subcommand is using the following environment variables:

- BIND_ADDRESS (optional): The address on which it will start listening for TCP connections (default value: localhost)
- BIND_PORT: The port on which it will start listening for TCP connections.
- EXPECTED_CLIENT_DATA (optional): If set, it will check that the request sends the same exact data.
- CHUNKS: How many chunks of data to write in the response.
- CHUNK_SIZE: The expected size of each written chunk of data. If it does not match the actual size of the written data, it will exit with the exit code 4.
- CHUNK_INTERVAL: The amount of time to wait in between chunks.`,
	Args: cobra.MaximumNArgs(0),
	Run:  main,
}

func main(cmd *cobra.Command, args []string) {
	bindAddress := os.Getenv("BIND_ADDRESS")
	if bindAddress == "" {
		bindAddress = "localhost"
	}
	bindPort := os.Getenv("BIND_PORT")
	addr, err := net.ResolveTCPAddr("tcp", net.JoinHostPort(bindAddress, bindPort))
	if err != nil {
		fmt.Printf("Error resolving: %v\n", err)
		os.Exit(1)
	}
	listener, err := net.ListenTCP("tcp", addr)
	if err != nil {
		fmt.Printf("Error listening: %v\n", err)
		os.Exit(1)
	}

	conn, err := listener.AcceptTCP()
	if err != nil {
		fmt.Printf("Error accepting connection: %v\n", err)
		os.Exit(1)
	}

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

	fmt.Println("Shutting down connection")

	// set linger timeout to flush buffers. This is the official way according to the go api docs. But
	// there are controversial discussions whether this value has any impact on most platforms
	// (compare https://codereview.appspot.com/95320043).
	conn.SetLinger(-1)

	// Flush the connection cleanly, following https://blog.netherlabs.nl/articles/2009/01/18/the-ultimate-so_linger-page-or-why-is-my-tcp-not-reliable:
	// 1. close write half of connection which sends a FIN packet
	// 2. give client some time to receive the FIN
	// 3. close the complete connection
	conn.CloseWrite()
	time.Sleep(rstAvoidanceDelay)
	conn.Close()

	fmt.Println("Done")
}

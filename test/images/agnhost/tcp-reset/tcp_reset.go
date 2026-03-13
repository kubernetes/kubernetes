/*
Copyright 2023 The Kubernetes Authors.

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

// A small utility to send RST on the TCP connection
// Ref: https://gosamples.dev/connection-reset-by-peer/

package tcpreset

import (
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"
)

// CmdTCPReset is used by agnhost Cobra.
var CmdTCPReset = &cobra.Command{
	Use:   "tcp-reset",
	Short: "Serves on a tcp port and RST the connections received",
	Long:  `Serves on a tcp port and RST the connections received.`,
	Args:  cobra.MaximumNArgs(0),
	Run:   main,
}

var (
	port int
)

func init() {
	CmdTCPReset.Flags().IntVar(&port, "port", 8080, "Port number.")
}

func main(cmd *cobra.Command, args []string) {

	listener, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Fatalf("Error from net.Listen(): %s", err)
	}
	go func() {
		for {
			conn, err := listener.Accept()
			if err != nil {
				log.Fatalf("Error from Accept(): %s", err)
			}
			// if there are still data in the buffer to read
			// and the connection is closed, it will send a RST.
			buf := make([]byte, 1)
			conn.Read(buf)
			conn.Close()
			log.Printf("TCP request from %s", conn.RemoteAddr().String())
		}
	}()

	log.Printf("Serving on port %d.\n", port)
	signals := make(chan os.Signal, 1)
	signal.Notify(signals, syscall.SIGTERM)
	sig := <-signals
	log.Printf("Shutting down after receiving signal: %s.\n", sig)
	log.Printf("Awaiting pod deletion.\n")

}

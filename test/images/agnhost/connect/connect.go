/*
Copyright 2019 The Kubernetes Authors.

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

package connect

import (
	"fmt"
	"net"
	"os"
	"strings"
	"syscall"
	"time"

	"github.com/ishidawataru/sctp"
	"github.com/spf13/cobra"
)

// CmdConnect is used by agnhost Cobra.
var CmdConnect = &cobra.Command{
	Use:   "connect [host:port]",
	Short: "Attempts a TCP, UDP or SCTP connection and returns useful errors",
	Long: `Tries to open a TCP, UDP or SCTP connection to the given host and port. On error it prints an error message prefixed with a specific fixed string that test cases can check for:

* UNKNOWN - Generic/unknown (non-network) error (eg, bad arguments)
* TIMEOUT - The connection attempt timed out
* DNS - An error in DNS resolution
* REFUSED - Connection refused
* OTHER - Other networking error (eg, "no route to host")`,
	Args: cobra.ExactArgs(1),
	Run:  main,
}

var (
	timeout  time.Duration
	protocol string
	udpData  string
)

func init() {
	CmdConnect.Flags().DurationVar(&timeout, "timeout", time.Duration(0), "Maximum time before returning an error")
	CmdConnect.Flags().StringVar(&protocol, "protocol", "tcp", "The protocol to use to perform the connection, can be tcp, udp or sctp")
	CmdConnect.Flags().StringVar(&udpData, "udp-data", "hostname", "The UDP payload send to the server")
}

func main(cmd *cobra.Command, args []string) {
	dest := args[0]
	switch protocol {
	case "", "tcp":
		connectTCP(dest, timeout)
	case "udp":
		connectUDP(dest, timeout, udpData)
	case "sctp":
		connectSCTP(dest, timeout)
	default:
		fmt.Fprint(os.Stderr, "Unsupported protocol\n", protocol)
		os.Exit(1)
	}
}

func connectTCP(dest string, timeout time.Duration) {
	// Redundantly parse and resolve the destination so we can return the correct
	// errors if there's a problem.
	if _, _, err := net.SplitHostPort(dest); err != nil {
		fmt.Fprintf(os.Stderr, "UNKNOWN: %v\n", err)
		os.Exit(1)
	}
	if _, err := net.ResolveTCPAddr("tcp", dest); err != nil {
		fmt.Fprintf(os.Stderr, "DNS: %v\n", err)
		os.Exit(1)
	}

	conn, err := net.DialTimeout("tcp", dest, timeout)
	if err == nil {
		conn.Close()
		os.Exit(0)
	}
	if opErr, ok := err.(*net.OpError); ok {
		if opErr.Timeout() {
			fmt.Fprintf(os.Stderr, "TIMEOUT\n")
			os.Exit(1)
		} else if syscallErr, ok := opErr.Err.(*os.SyscallError); ok {
			if syscallErr.Err == syscall.ECONNREFUSED {
				fmt.Fprintf(os.Stderr, "REFUSED\n")
				os.Exit(1)
			}
		}
	}

	fmt.Fprintf(os.Stderr, "OTHER: %v\n", err)
	os.Exit(1)
}

func connectSCTP(dest string, timeout time.Duration) {
	addr, err := sctp.ResolveSCTPAddr("sctp", dest)
	if err != nil {
		fmt.Fprintf(os.Stderr, "DNS: %v\n", err)
		os.Exit(1)
	}

	timeoutCh := time.After(timeout)
	errCh := make(chan error)

	go func() {
		conn, err := sctp.DialSCTP("sctp", nil, addr)
		if err == nil {
			conn.Close()
		}
		errCh <- err
	}()

	select {
	case err := <-errCh:
		if err != nil {
			fmt.Fprintf(os.Stderr, "OTHER: %v\n", err)
			os.Exit(1)
		}
	case <-timeoutCh:
		fmt.Fprint(os.Stderr, "TIMEOUT\n")
		os.Exit(1)
	}
}

func connectUDP(dest string, timeout time.Duration, data string) {
	var (
		readBytes int
		buf       = make([]byte, 1024)
	)

	if _, err := net.ResolveUDPAddr("udp", dest); err != nil {
		fmt.Fprintf(os.Stderr, "DNS: %v\n", err)
		os.Exit(1)
	}

	conn, err := net.Dial("udp", dest)
	if err != nil {
		fmt.Fprintf(os.Stderr, "OTHER: %v\n", err)
		os.Exit(1)
	}

	if timeout > 0 {
		if err = conn.SetDeadline(time.Now().Add(timeout)); err != nil {
			fmt.Fprintf(os.Stderr, "OTHER: %v\n", err)
			os.Exit(1)
		}
	}

	if _, err = conn.Write([]byte(fmt.Sprintf("%s\n", data))); err != nil {
		parseUDPErrorAndExit(err)
	}

	if readBytes, err = conn.Read(buf); err != nil {
		parseUDPErrorAndExit(err)
	}

	// ensure the response from UDP server
	if readBytes == 0 {
		fmt.Fprintf(os.Stderr, "OTHER: No data received from the server. Cannot guarantee the server received the request.\n")
		os.Exit(1)
	}
}

func parseUDPErrorAndExit(err error) {
	neterr, ok := err.(net.Error)
	if ok && neterr.Timeout() {
		fmt.Fprintf(os.Stderr, "TIMEOUT: %v\n", err)
	} else if strings.Contains(err.Error(), "connection refused") {
		fmt.Fprintf(os.Stderr, "REFUSED: %v\n", err)
	} else {
		fmt.Fprintf(os.Stderr, "UNKNOWN: %v\n", err)
	}
	os.Exit(1)
}

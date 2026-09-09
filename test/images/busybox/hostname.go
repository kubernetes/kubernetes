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

package main

import (
	"flag"
	"fmt"
	"log"
	"net"
	"os"
)

func getOutboundIP() net.IP {
	conn, err := net.Dial("udp", "8.8.8.8:80")
	if err != nil {
		log.Fatal(err)
	}
	defer conn.Close()
	localAddr := conn.LocalAddr().(*net.UDPAddr)
	return localAddr.IP
}

func getHostname() (string, error) {
	return os.Hostname()
}

func main() {
	flagIP := flag.Bool("i", false, "display the IP address of the host")
	flagFQDN := flag.Bool("f", false, "display the fully qualified domain name")
	flagFQDNLong := flag.Bool("fqdn", false, "display the fully qualified domain name")
	flagLong := flag.Bool("long", false, "display the fully qualified domain name")
	flag.Parse()
	switch {
	case *flagIP:
		ip := getOutboundIP()
		fmt.Print(ip.String())
	case *flagFQDN || *flagFQDNLong || *flagLong:
		fqdn, err := getFQDN()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print(fqdn)
	default:
		hostname, err := getHostname()
		if err != nil {
			log.Fatal(err)
		}
		fmt.Print(hostname)
	}
}

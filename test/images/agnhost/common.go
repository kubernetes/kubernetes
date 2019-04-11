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

package main

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"syscall"

	"github.com/spf13/cobra"
)

func printDNSSuffixList(cmd *cobra.Command, args []string) {
	dnsSuffixList := getDNSSuffixList()
	fmt.Println(strings.Join(dnsSuffixList, ","))
}

func printDNSServerList(cmd *cobra.Command, args []string) {
	dnsServerList := getDNSServerList()
	fmt.Println(strings.Join(dnsServerList, ","))
}

func printHostsFile(cmd *cobra.Command, args []string) {
	fmt.Println(readFile(etcHostsFile))
}

func pause(cmd *cobra.Command, args []string) {
	sigCh := make(chan os.Signal)
	done := make(chan int, 1)
	signal.Notify(sigCh, syscall.SIGINT)
	signal.Notify(sigCh, syscall.SIGTERM)
	signal.Notify(sigCh, syscall.SIGKILL)
	go func() {
		sig := <-sigCh
		switch sig {
		case syscall.SIGINT:
			done <- 1
			os.Exit(1)
		case syscall.SIGTERM:
			done <- 2
			os.Exit(2)
		case syscall.SIGKILL:
			done <- 0
			os.Exit(0)
		}
	}()
	result := <-done
	fmt.Printf("exiting %d\n", result)
}

func readFile(fileName string) string {
	fileData, err := ioutil.ReadFile(fileName)
	if err != nil {
		panic(err)
	}

	return string(fileData)
}

func runCommand(name string, arg ...string) string {
	var out bytes.Buffer
	cmd := exec.Command(name, arg...)
	cmd.Stdout = &out

	err := cmd.Run()
	if err != nil {
		panic(err)
	}

	return strings.TrimSpace(out.String())
}

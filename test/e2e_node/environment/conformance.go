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

// Build the binary with `go build conformance.go`, then run the conformance binary on a node candidate.  If compiled
// on a non-linux machine, must be cross compiled for the host.
package main

import (
	"flag"
	"fmt"
	"net"
	"os/exec"
	"regexp"
	"strings"

	"errors"
	"os"
)

const success = "\033[0;32mSUCCESS\033[0m"
const failed = "\033[0;31mFAILED\033[0m"
const notConfigured = "\033[0;34mNOT CONFIGURED\033[0m"
const skipped = "\033[0;34mSKIPPED\033[0m"

var checkFlag = flag.String(
	"check", "all", "what to check for conformance.  One or more of all,container-runtime,daemons,dns,firewall,kernel")

func init() {
	// Set this to false to undo util/logs.go settings it to true.  Prevents cadvisor log spam.
	// Remove this once util/logs.go stops setting the flag to true.
	flag.Set("logtostderr", "false")
}

// TODO: Should we write an e2e test for this?
func main() {
	flag.Parse()
	o := strings.Split(*checkFlag, ",")
	errs := check(o...)
	if len(errs) > 0 {
		os.Exit(1)
	} else {
		os.Exit(0)
	}
}

// check returns errors found while checking the provided components.  Will prevent errors to stdout.
func check(options ...string) []error {
	errs := []error{}
	for _, c := range options {
		switch c {
		case "all":
			errs = appendNotNil(errs, kernel())
			errs = appendNotNil(errs, daemons())
			errs = appendNotNil(errs, firewall())
			errs = appendNotNil(errs, dns())
		case "daemons":
			errs = appendNotNil(errs, daemons())
		case "dns":
			errs = appendNotNil(errs, dns())
		case "firewall":
			errs = appendNotNil(errs, firewall())
		case "kernel":
			errs = appendNotNil(errs, kernel())
		default:
			fmt.Printf("Unrecognized option %s\n", c)
			errs = append(errs, fmt.Errorf("Unrecognized option %s", c))
		}
	}
	return errs
}

const kubeletClusterDNSRegexStr = `\/kubelet.*--cluster-dns=(\S+) `
const kubeletClusterDomainRegexStr = `\/kubelet.*--cluster-domain=(\S+)`

// dns checks that cluster dns has been properly configured and can resolve the kubernetes.default service
func dns() error {
	dnsRegex, err := regexp.Compile(kubeletClusterDNSRegexStr)
	if err != nil {
		// This should never happen and can only be fixed by changing the code
		panic(err)
	}
	domainRegex, err := regexp.Compile(kubeletClusterDomainRegexStr)
	if err != nil {
		// This should never happen and can only be fixed by changing the code
		panic(err)
	}

	h, err := net.LookupHost("kubernetes.default")
	if err == nil {
		return printSuccess("Dns Check (Optional): %s", success)
	}
	if len(h) > 0 {
		return printSuccess("Dns Check (Optional): %s", success)
	}

	kubecmd, err := exec.Command("ps", "aux").CombinedOutput()
	if err != nil {
		// Executing ps aux shouldn't have failed
		panic(err)
	}

	// look for the dns flag and parse the value
	dns := dnsRegex.FindStringSubmatch(string(kubecmd))
	if len(dns) < 2 {
		return printSuccess(
			"Dns Check (Optional): %s No hosts resolve to kubernetes.default.  kubelet will need to set "+
				"--cluster-dns and --cluster-domain when run", notConfigured)
	}

	// look for the domain flag and parse the value
	domain := domainRegex.FindStringSubmatch(string(kubecmd))
	if len(domain) < 2 {
		return printSuccess(
			"Dns Check (Optional): %s No hosts resolve to kubernetes.default.  kubelet will need to set "+
				"--cluster-dns and --cluster-domain when run", notConfigured)
	}

	// do a lookup with the flags the kubelet is running with
	nsArgs := []string{"-q=a", fmt.Sprintf("kubernetes.default.%s", domain[1]), dns[1]}
	if err = exec.Command("nslookup", nsArgs...).Run(); err != nil {
		// Mark this as failed since there was a clear intention to set it up, but it is done so improperly
		return printError(
			"Dns Check (Optional): %s No hosts resolve to kubernetes.default  kubelet found, but cannot resolve "+
				"kubernetes.default using nslookup %s error: %v", failed, strings.Join(nsArgs, " "), err)
	}

	// Can resolve kubernetes.default using the kubelete dns and domain values
	return printSuccess("Dns Check (Optional): %s", success)
}

const cmdlineCGroupMemory = `cgroup_enable=memory`

// kernel checks that the kernel has been configured correctly to support the required cgroup features
func kernel() error {
	cmdline, err := os.ReadFile("/proc/cmdline")
	if err != nil {
		return printError("Kernel Command Line Check %s: Could not check /proc/cmdline", failed)
	}
	if !strings.Contains(string(cmdline), cmdlineCGroupMemory) {
		return printError("Kernel Command Line Check %s: cgroup_enable=memory not enabled in /proc/cmdline", failed)
	}
	return printSuccess("Kernel Command Line %s", success)
}

const iptablesInputRegexStr = `Chain INPUT \(policy DROP\)`
const iptablesForwardRegexStr = `Chain FORWARD \(policy DROP\)`

// firewall checks that iptables does not have common firewall rules setup that would disrupt traffic
func firewall() error {
	out, err := exec.Command("iptables", "-L", "INPUT").CombinedOutput()
	if err != nil {
		return printSuccess("Firewall IPTables Check %s: Could not run iptables", skipped)
	}
	inputRegex, err := regexp.Compile(iptablesInputRegexStr)
	if err != nil {
		// This should never happen and can only be fixed by changing the code
		panic(err)
	}
	if inputRegex.Match(out) {
		return printError("Firewall IPTables Check %s: Found INPUT rule matching %s", failed, iptablesInputRegexStr)
	}

	// Check GCE forward rules
	out, err = exec.Command("iptables", "-L", "FORWARD").CombinedOutput()
	if err != nil {
		return printSuccess("Firewall IPTables Check %s: Could not run iptables", skipped)
	}
	forwardRegex, err := regexp.Compile(iptablesForwardRegexStr)
	if err != nil {
		// This should never happen and can only be fixed by changing the code
		panic(err)
	}
	if forwardRegex.Match(out) {
		return printError("Firewall IPTables Check %s: Found FORWARD rule matching %s", failed, iptablesInputRegexStr)
	}

	return printSuccess("Firewall IPTables Check %s", success)
}

// daemons checks that the required node programs are running: kubelet and kube-proxy
func daemons() error {
	if exec.Command("pgrep", "-f", "kubelet").Run() != nil {
		return printError("Daemon Check %s: kubelet process not found", failed)
	}

	if exec.Command("pgrep", "-f", "kube-proxy").Run() != nil {
		return printError("Daemon Check %s: kube-proxy process not found", failed)
	}

	return printSuccess("Daemon Check %s", success)
}

// printError provides its arguments to print a format string to the console (newline terminated) and returns an
// error with the same string
func printError(s string, args ...interface{}) error {
	es := fmt.Sprintf(s, args...)
	fmt.Println(es)
	return errors.New(es)
}

// printSuccess provides its arguments to print a format string to the console (newline terminated) and returns nil
func printSuccess(s string, args ...interface{}) error {
	fmt.Println(fmt.Sprintf(s, args...))
	return nil
}

// appendNotNil appends err to errs iff err is not nil
func appendNotNil(errs []error, err error) []error {
	if err != nil {
		return append(errs, err)
	}
	return errs
}

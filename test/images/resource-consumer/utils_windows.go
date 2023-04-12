//go:build windows
// +build windows

/*
Copyright 2021 The Kubernetes Authors.

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
	"log"
	"os/exec"
	"strconv"
	"time"
)

var (
	consumeCPUBinary = "./consume-cpu/consume-cpu.exe"
	consumeMemBinary = "testlimit.exe"
)

// ConsumeMem consumes a given number of megabytes for the specified duration.
func ConsumeMem(megabytes int, durationSec int) {
	log.Printf("ConsumeMem megabytes: %v, durationSec: %v", megabytes, durationSec)
	megabytesString := strconv.Itoa(megabytes)
	durationSecString := strconv.Itoa(durationSec)
	// creating new consume memory process
	consumeMem := exec.Command(consumeMemBinary, "-accepteula", "-d", megabytesString, "-e", "0", durationSecString, "-c", "1")
	err := consumeMem.Start()
	if err != nil {
		log.Printf("Error while consuming memory: %v", err)
		return
	}

	// testlimit does not end after durationSec elapsed, so we'll stop it ourselves.
	time.AfterFunc(time.Duration(durationSec)*time.Second, func() {
		if err := consumeMem.Process.Kill(); err != nil {
			log.Printf("Could not kill testlimit process! Error: %v", err)
		}
	})
}

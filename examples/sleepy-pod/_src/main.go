/*
Copyright 2015 Google Inc. All rights reserved.

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
	"fmt"
	"io/ioutil"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func handle_signals(logFilePath string) {
	c := make(chan os.Signal, 1)
	signal.Notify(c, os.Interrupt)
	signal.Notify(c, syscall.SIGTERM)
	sig := <-c
	if sig == os.Interrupt {
		fmt.Printf("sleepy was interrupted. sigh.. Writing to log at %s\n", logFilePath)
		ioutil.WriteFile(logFilePath, []byte("\nsleepy was interrupted. sigh.."), 0644)
	} else {
		fmt.Printf("sleepy was killed. sob.. Writing to log at %s\n", logFilePath)
		ioutil.WriteFile(logFilePath, []byte("\nsleepy was killed. sob.."), 0644)
	}
	os.Exit(1)
}

func main() {
	terminationMessagePath := "/dev/termination-log"
	if len(os.Args) >= 2 {
		terminationMessagePath = os.Args[1]
	}

	go handle_signals(terminationMessagePath)
	for true {
		fmt.Printf("sleepy with process id '%d' is sleeping...yawn..interrupt me..\n", os.Getpid())
		time.Sleep(time.Millisecond * 1000)
	}
	fmt.Println("sleepy should never have stopped.")
}

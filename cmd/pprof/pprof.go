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

package pprof

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/pprof"
	"time"
)

func init() {
	go run()
}

func run() {
	profileDir := os.Getenv("PROFILE_DIR")
	if profileDir == "" {
		return
	}
	if err := os.MkdirAll(profileDir, os.FileMode(0755)); err != nil {
		fmt.Println(err)
		return
	}

	_, command := filepath.Split(os.Args[0])

	for {
		filename := command + "." + time.Now().Format("15_04_05")
		cpufile := filename + ".cpu.pprof"
		memfile := filename + ".mem.pprof"

		func() {
			cpu, err := os.Create(filepath.Join(profileDir, cpufile))
			if err != nil {
				fmt.Println("could not create CPU profile: ", err)
				return
			}
			defer cpu.Close()

			mem, err := os.Create(filepath.Join(profileDir, memfile))
			if err != nil {
				fmt.Println("could not create memory profile: ", err)
				return
			}
			defer mem.Close()

			if err := pprof.StartCPUProfile(cpu); err != nil {
				fmt.Println("could not create CPU profile: ", err)
				return
			}
			defer func() {
				runtime.GC() // get up-to-date statistics
				if err := pprof.WriteHeapProfile(mem); err != nil {
					fmt.Println("could not write memory profile: ", err)
					return
				}
			}()
			defer pprof.StopCPUProfile()
			time.Sleep(time.Minute)
		}()
	}
}

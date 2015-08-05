/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package profiler

import (
	"fmt"
	"io/ioutil"
	"os"
	"os/exec"
	"os/signal"
	"runtime"
	"runtime/pprof"
	"sync"
	"syscall"
	"time"

	"github.com/golang/glog"
)

// Profiler is anything that can run/halt a profiler.
type Profiler interface {
	Run()
	Halt()
}

// pprofProfiler implements Profiler.
type pprofProfiler struct {
	dirName string
	prefix  string
	count   int
	mutex   sync.Mutex
	running bool
	stopCh  chan struct{}
	once    sync.Once
}

// NewProfiler creates a new Profiler.
func NewProfiler(prefix string) Profiler {
	return &pprofProfiler{prefix: prefix, count: 0, running: false, stopCh: make(chan struct{})}
}

// Run starts a background thread that listens for SIGUSR1.
func (p *pprofProfiler) Run() {
	p.once.Do(p.run)
}

func (p *pprofProfiler) run() {
	// Multiple SIGUSR1's will get dropped.
	signalChan := make(chan os.Signal, 1)
	signal.Notify(signalChan, syscall.SIGUSR1)

	glog.Infof("Starting profile manager. Kill -10 to start/stop profiling.")

	// Signal handler.
	go func() {
		for {
			// Calls to start/stop the profiler are not thread safe, so they're restricted
			// to this thread.
			select {
			case <-signalChan:
				if p.isRunning() {
					p.stop()
				} else {
					p.start()
				}

			// This 2 phase stopping is required so the caller can signal a Halt and wait
			// for the final tarball process to complete before exiting.
			case <-p.stopCh:
				p.stop()
				p.stopCh <- struct{}{}
				break
			}
		}
	}()
}

// Halt halts a running profiler.
func (p *pprofProfiler) Halt() {
	if !p.isRunning() {
		return
	}

	p.stopCh <- struct{}{}
	<-p.stopCh
}

// start starts a cpu profile, and dumps a heap profile
func (p *pprofProfiler) start() {

	if p.isRunning() {
		glog.V(3).Infof("Profiler already running, start call will no-op")
		return
	}

	dirName, err := ioutil.TempDir("", p.prefix)
	if err != nil {
		glog.Errorf("Failed to start profile manager, cannot create tempdir")
		return
	}
	p.dirName = dirName
	glog.Infof("Recording new profile in %v", p.dirName)

	f, err := ioutil.TempFile(p.dirName, p.tag("cpu"))
	if err != nil {
		glog.Infof("Failed to create tmp file in dir %v", p.dirName)
		return
	}

	pprof.StartCPUProfile(f)
	HeapProfile(p.dirName, p.tag("start_heap"))
	p.setRunning(true)
}

// stop stops a previously started cpu profile, dumps a heap profile,
// and tars the entire profile subdirectory.
func (p *pprofProfiler) stop() {

	if !p.isRunning() {
		glog.V(3).Infof("Profiler not running, stop call will no-op")
		return
	}

	defer p.setRunning(false)

	pprof.StopCPUProfile()
	HeapProfile(p.dirName, p.tag("end_heap"))

	// The profile is useless without the symbols and executable.
	copyExecutable(p.dirName)
	tarName := fmt.Sprintf("%v_%v.tar", p.prefix, time.Now().Format("20060102150405"))

	if err := tar(p.dirName, tarName); err != nil {
		glog.Infof("Failed to dump profile: %v, profile left in %v", err, p.dirName)
	} else {
		rm(p.dirName)
	}
}

// HeapProfile dumps a heap profile
func HeapProfile(dirName, prefix string) {

	f, err := ioutil.TempFile(dirName, prefix)
	if err != nil {
		glog.Infof("Failed to create tmp file for profiles in dir %v", dirName)
		return
	}

	pprof.WriteHeapProfile(f)
	f.Close()
}

func (p *pprofProfiler) isRunning() bool {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	return p.running
}

func (p *pprofProfiler) setRunning(running bool) {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	// Increment the profile count everytime a profile is stopped
	if !running {
		p.count++
	}
	p.running = running
}

func (p *pprofProfiler) tag(name string) string {
	return fmt.Sprintf("%s_%s_%d", p.prefix, name, p.count)
}

// copyExecutable copies the current executable to the profile directory.
func copyExecutable(destDir string) {
	switch plat := runtime.GOOS; plat {
	case "linux":
		if exePath, err := os.Readlink("/proc/self/exe"); err == nil {
			if err := exec.Command("cp", exePath, destDir).Run(); err == nil {
				return
			}
		}
	default:
		glog.Infof("Cannot copy binary on platform %v", plat)
	}
	glog.Infof("Please copy current binary by hand.")
	return
}

// tar creates a tar file of the files in the given directory.
func tar(destDir, tarfile string) error {
	tarfile = "/tmp/" + tarfile
	if err := exec.Command("tar", "czf", tarfile, destDir).Run(); err != nil {
		return err
	}
	glog.Infof("Profile tarred to %v", tarfile)
	return nil
}

// rm removes given directory.
func rm(destDir string) {
	if destDir == "" {
		return
	}
	if err := exec.Command("rm", "-rf", destDir).Run(); err != nil {
		glog.Errorf("Failed to remove profile dir %v", destDir)
		return
	}
	glog.Infof("Removed profile dir %v", destDir)
}

// Copyright 2018 Microsoft Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cmd

import (
	"bufio"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"time"
)

type work struct {
	filename string
	options  []string
}

func autorestCommand(file string, options []string) *exec.Cmd {
	options = append(options, file)
	c := exec.Command("autorest", options...)
	return c
}

func worker(id int, jobs <-chan work, results chan<- error) {
	for work := range jobs {
		start := time.Now()
		c := autorestCommand(work.filename, work.options)
		vprintf("worker %d is starting on file %s\nparameters: %v\n", id, work.filename, c.Args)
		output, err := c.CombinedOutput()
		if err == nil {
			vprintf("worker %d has done with file %s (%v)\n", id, work.filename, time.Since(start))
		} else {
			printf("worker %d fails with file %s (%v), error messages:\n%v\n", id, work.filename, time.Since(start), string(output))
		}
		results <- err
	}
}

func startCmd(c *exec.Cmd) error {
	stdout, err := c.StdoutPipe()
	if err != nil {
		return fmt.Errorf("failed to get stdout pipe: %v", err)
	}
	scanner := bufio.NewScanner(stdout)
	go func() {
		for scanner.Scan() {
			printf("> %s\n", scanner.Text())
		}
	}()
	stderr, err := c.StderrPipe()
	if err != nil {
		return fmt.Errorf("failed to get stderr pipe: %v", err)
	}
	scanner = bufio.NewScanner(stderr)
	go func() {
		for scanner.Scan() {
			printf("> %s\n", scanner.Text())
		}
	}()
	return c.Start()
}

func selectFilesWithName(path string, name string) ([]string, error) {
	var files []string
	err := filepath.Walk(path, func(p string, info os.FileInfo, err error) error {
		if !info.IsDir() && info.Name() == name {
			files = append(files, p)
		}
		return nil
	})
	return files, err
}

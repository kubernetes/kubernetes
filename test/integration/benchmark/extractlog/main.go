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
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
)

func main() {
	err := extractRawLog(os.Stdin)
	if err != nil {
		panic(err)
	}
}

// A json log entry contains keys such as "Time", "Action", "Package" and "Output".
// We are only interested in "Output", which is the raw log.
type jsonLog struct {
	Output string `json:"output,omitempty"`
}

// jsonToRawLog converts a single line of json formatted log to raw log.
// If there is an error, it returns the original input.
func jsonToRawLog(line string) (string, error) {
	var log jsonLog
	if err := json.Unmarshal([]byte(line), &log); err != nil {
		return line, err
	}
	return log.Output, nil
}

func extractRawLog(r io.Reader) error {
	scan := bufio.NewScanner(r)
	for scan.Scan() {
		l, _ := jsonToRawLog(scan.Text())
		// Print the raw log to stdout.
		fmt.Println(l)
	}
	if err := scan.Err(); err != nil {
		return err
	}
	return nil
}

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package iptables

import (
	"bytes"
	"errors"
	"fmt"
	"os/exec"
)

// TODO: we use os.exec here for some things because we need to be able to write to stdin (rather than util/exec).
// We should possibly add support to do this from util/exec instead.
// Also note this: https://github.com/golang/go/issues/7990
// when messing with the stdin piping.
func runCmdWithStdin(command string, args []string, input []byte) ([]byte, error) {
	cmd := exec.Command(command, args...)
	stdin, err := cmd.StdinPipe()
	var b bytes.Buffer
	if err != nil {
		return b.Bytes(), err
	}
	go func() {
		stdin.Write(input)
		stdin.Close()
	}()
	cmd.Stdout = &b
	cmd.Stderr = &b
	err = cmd.Run()
	return b.Bytes(), err
}

// Restore runs iptables restore passing args as arugments and passing rules via stdin.
// The error returned if any will also have the output appended in parentheses.
func Restore(args []string, rules []byte) error {
	b, err := runCmdWithStdin("iptables-restore", args, rules)
	if err != nil {
		return errors.New(fmt.Sprintf("%v (%s)", err, b))
	}
	return nil
}

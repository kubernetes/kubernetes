/*
Copyright 2018 The Kubernetes Authors.

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

package exec_test

import (
	"fmt"
	"io/ioutil"

	"k8s.io/utils/exec"
)

func ExampleNew_stderrPipe() {
	cmd := exec.New().Command("/bin/sh", "-c", "echo 'We can read from stderr via pipe!' >&2")

	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		panic(err)
	}

	stderr := make(chan []byte)
	go func() {
		b, err := ioutil.ReadAll(stderrPipe)
		if err != nil {
			panic(err)
		}
		stderr <- b
	}()

	if err := cmd.Start(); err != nil {
		panic(err)
	}

	received := <-stderr

	if err := cmd.Wait(); err != nil {
		panic(err)
	}

	fmt.Println(string(received))
	// Output: We can read from stderr via pipe!
}

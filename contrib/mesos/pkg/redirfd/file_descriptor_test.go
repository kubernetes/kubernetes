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

package redirfd

import (
	"testing"

	. "github.com/onsi/gomega"
)

func TestParseFileDescriptor(t *testing.T) {
	RegisterTestingT(t)

	valid := map[string]FileDescriptor{
		"-1": InvalidFD,
		"0":  Stdin,
		"1":  Stdout,
		"2":  Stderr,
		"3":  FileDescriptor(3),
	}

	for input, expected := range valid {
		fd, err := ParseFileDescriptor(input)
		Expect(err).ToNot(HaveOccurred(), "Input: '%s'", input)
		Expect(fd).To(Equal(expected), "Input: '%s'", input)
	}

	invalid := []string{
		"a",
		" 1",
		"blue",
		"stderr",
		"STDERR",
	}

	for _, input := range invalid {
		_, err := ParseFileDescriptor(input)
		Expect(err).To(HaveOccurred(), "Input: '%s'", input)
	}
}

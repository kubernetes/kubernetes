/* Copyright 2016 The Bazel Authors. All rights reserved.

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
	"bytes"
	"io/ioutil"
	"os"
	"os/exec"

	"github.com/bazelbuild/bazel-gazelle/internal/config"
	bf "github.com/bazelbuild/buildtools/build"
)

func diffFile(c *config.Config, file *bf.File, path string) error {
	oldContents, err := ioutil.ReadFile(file.Path)
	if err != nil {
		oldContents = nil
	}
	newContents := bf.Format(file)
	if bytes.Equal(oldContents, newContents) {
		return nil
	}
	f, err := ioutil.TempFile("", c.DefaultBuildFileName())
	if err != nil {
		return err
	}
	f.Close()
	defer os.Remove(f.Name())
	if err := ioutil.WriteFile(f.Name(), newContents, 0666); err != nil {
		return err
	}
	cmd := exec.Command("diff", "-u", "--new-file", path, f.Name())
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	err = cmd.Run()
	if _, ok := err.(*exec.ExitError); ok {
		// diff returns non-zero when files are different. This is not an error.
		return nil
	}
	return err
}

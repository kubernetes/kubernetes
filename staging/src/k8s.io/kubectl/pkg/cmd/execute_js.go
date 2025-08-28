//go:build js
// +build js

/*
Copyright 2016 The Kubernetes Authors.

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

package cmd

import "os"
import "fmt"

func execute(executablePath string, cmdArgs, environment []string) error {
	fmt.Println("bonjour de la")
	cmd := Command(executablePath, cmdArgs...)
	fmt.Println("bonjour d'ici")

	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Stdin = os.Stdin
	cmd.Env = environment

	err := cmd.Run()

	if err == nil {
		os.Exit(0)
	}
	return err
}

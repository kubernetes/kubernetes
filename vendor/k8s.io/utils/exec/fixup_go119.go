//go:build go1.19
// +build go1.19

/*
Copyright 2022 The Kubernetes Authors.

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

package exec

import (
	"errors"
	osexec "os/exec"
)

// maskErrDotCmd reverts the behavior of osexec.Cmd to what it was before go1.19
// specifically set the Err field to nil (LookPath returns a new error when the file
// is resolved to the current directory.
func maskErrDotCmd(cmd *osexec.Cmd) *osexec.Cmd {
	cmd.Err = maskErrDot(cmd.Err)
	return cmd
}

func maskErrDot(err error) error {
	if err != nil && errors.Is(err, osexec.ErrDot) {
		return nil
	}
	return err
}

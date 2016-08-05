// +build windows

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
	"fmt"
	"os"
)

type RedirectMode int

const (
	Read           RedirectMode = iota // open file for reading
	Write                              // open file for writing, truncating if it exists
	Update                             // open file for read & write
	Append                             // open file for append, create if it does not exist
	AppendExisting                     // open file for append, do not create if it does not already exist
	WriteNew                           // open file for writing, creating it, failing if it already exists
)

func (mode RedirectMode) Redirect(nonblock, changemode bool, fd FileDescriptor, name string) (*os.File, error) {
	return nil, fmt.Errorf("Redirect(%s, %s, %d, \"%s\") not supported on windows", nonblock, changemode, fd, name)
}

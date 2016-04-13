/*
Copyright 2015 The go4 Authors

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

// Package osutil contains os level functions.
package osutil // import "go4.org/osutil"

// capture executable on package init to work around various os issues if
// captured after executable has been renamed.
var execPath, execError = executable()

// Executable returns the path name for the executable that starts the
// current process. The result is the path that was used to start the
// current process, but there is no guarantee that the path is still
// pointing to the correct executable.
//
// OpenBSD is currently unsupported.
func Executable() (string, error) {
	return execPath, execError
}

/*
Copyright 2017 The Kubernetes Authors.

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

package securitycontext

import (
	"fmt"
)

// DockerLabelUser returns the fragment of a Docker security opt that
// describes the SELinux user. Note that strictly speaking this is not
// actually the name of the security opt, but a fragment of the whole key-
// value pair necessary to set the opt.
func DockerLabelUser(separator rune) string {
	return fmt.Sprintf("label%cuser", separator)
}

// DockerLabelRole returns the fragment of a Docker security opt that
// describes the SELinux role. Note that strictly speaking this is not
// actually the name of the security opt, but a fragment of the whole key-
// value pair necessary to set the opt.
func DockerLabelRole(separator rune) string {
	return fmt.Sprintf("label%crole", separator)
}

// DockerLabelType returns the fragment of a Docker security opt that
// describes the SELinux type. Note that strictly speaking this is not
// actually the name of the security opt, but a fragment of the whole key-
// value pair necessary to set the opt.
func DockerLabelType(separator rune) string {
	return fmt.Sprintf("label%ctype", separator)
}

// DockerLabelLevel returns the fragment of a Docker security opt that
// describes the SELinux level. Note that strictly speaking this is not
// actually the name of the security opt, but a fragment of the whole key-
// value pair necessary to set the opt.
func DockerLabelLevel(separator rune) string {
	return fmt.Sprintf("label%clevel", separator)
}

// DockerLaelDisable returns the Docker security opt that disables SELinux for
// the container.
func DockerLabelDisable(separator rune) string {
	return fmt.Sprintf("label%cdisable", separator)
}

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

package platform

// shared constants for init
const (
	InitErrorNoSuchUser               = "no such user %q: %v"
	InitErrorUserIsRoot               = "user is root %q"
	InitErrorNoHomeDir                = "cannot obtain the home directory for user %q"
	InitErrorAtoiString               = "cannot Atoi() UID/GID string %q: %v"
	InitErrorCannotCreateDir          = "cannot create %q: %v"
	InitErrorCannotChown              = "cannot chown %q: %v"
	InitErrorCannotOpenFileForReading = "cannot open file for reading %q: %v"
	InitErrorCannotOpenFileForWriting = "cannot open file for writing %q: %v"
	InitErrorCannotIoCopy             = "cannot io.Copy() the configuration: %v"

	InitMessageCopyingCredentials = "[init] Copying administrator credentials to %q from %q\n"

	InitPathKube   = "/.kube"
	InitPathConfig = "/config"
)

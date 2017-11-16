/*
Copyright 2014 The Kubernetes Authors.

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

package strings

import (
	"strings"
)

// EscapePluginName converts a plugin name in the format
// vendor/pluginname into a proper ondisk vendor~pluginname plugin directory
// format.
func EscapePluginName(in string) string {
	return strings.Replace(in, "/", "~", -1)
}

// UnescapePluginName converts a plugin directory name in the format
// vendor~pluginname into a proper vendor/pluginname.
func UnescapePluginName(in string) string {
	return strings.Replace(in, "~", "/", -1)
}

// EscapeQualifiedNameForDisk converts a plugin name, which might contain a / into a
// string that is safe to use on-disk.  This assumes that the input has already
// been validates as a qualified name.  we use "~" rather than ":" here in case
// we ever use a filesystem that doesn't allow ":".
func EscapeQualifiedNameForDisk(in string) string {
	return strings.Replace(in, "/", "~", -1)
}

// UnescapeQualifiedNameForDisk converts an escaped plugin name (as per EscapeQualifiedNameForDisk)
// back to its normal form.  This assumes that the input has already been
// validates as a qualified name.
func UnescapeQualifiedNameForDisk(in string) string {
	return strings.Replace(in, "~", "/", -1)
}

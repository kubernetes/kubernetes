/*
Copyright 2020 The Kubernetes Authors.

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

package userdir

import (
	"os"
	"runtime"
)

// UserCacheDir returns the default root directory to use for user-specific
// cached data. Users should create their own application-specific subdirectory
// within this one and use that.
//
// It returns $XDG_CACHE_HOME if not empty, otherwise:
// On Unix systems, it returns $HOME/.cache.
// On Darwin, it returns $HOME/Library/Caches.
// On Windows, it returns %LocalAppData%.
// On Plan 9, it returns $home/lib/cache.
//
// If the location cannot be determined (for example, $HOME is not defined),
// then it will return an empty string.
func UserCacheDir() string {
	cacheHome := os.Getenv("XDG_CACHE_HOME")
	if len(cacheHome) != 0 {
		return cacheHome
	}

	// The below code is mainly copied from `os.UserCacheDir`.

	// The reason why not to use `os.UserCacheDir` here is that user may want to override the cache dir by setting the
	// `XDG_CACHE_HOME` environment variable.

	switch runtime.GOOS {
	case "windows":
		return os.Getenv("LocalAppData")

	case "darwin":
		dir := os.Getenv("HOME")
		if len(dir) == 0 {
			return ""
		}
		return dir + "/Library/Caches"

	case "plan9":
		dir := os.Getenv("home")
		if len(dir) == 0 {
			return ""
		}
		return dir + "/lib/cache"

	default: // Unix
		dir := os.Getenv("HOME")
		if len(dir) == 0 {
			return ""
		}
		return dir + "/.cache"
	}
}

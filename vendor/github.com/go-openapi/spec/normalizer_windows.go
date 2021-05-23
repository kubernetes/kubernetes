// -build windows

// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package spec

import (
	"net/url"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// absPath makes a file path absolute and compatible with a URI path component
//
// The parameter must be a path, not an URI.
func absPath(in string) string {
	// NOTE(windows): filepath.Abs exhibits a special behavior on windows for empty paths.
	// See https://github.com/golang/go/issues/24441
	if in == "" {
		in = "."
	}

	anchored, err := filepath.Abs(in)
	if err != nil {
		specLogger.Printf("warning: could not resolve current working directory: %v", err)
		return in
	}

	pth := strings.ReplaceAll(strings.ToLower(anchored), `\`, `/`)
	if !strings.HasPrefix(pth, "/") {
		pth = "/" + pth
	}

	return path.Clean(pth)
}

// repairURI tolerates invalid file URIs with common typos
// such as 'file://E:\folder\file', that break the regular URL parser.
//
// Adopting the same defaults as for unixes (e.g. return an empty path) would
// result into a counter-intuitive result for that case (e.g. E:\folder\file is
// eventually resolved as the current directory). The repair will detect the missing "/".
//
// Note that this only works for the file scheme.
func repairURI(in string) (*url.URL, string) {
	const prefix = fileScheme + "://"
	if !strings.HasPrefix(in, prefix) {
		// giving up: resolve to empty path
		u, _ := url.Parse("")

		return u, ""
	}

	// attempt the repair, stripping the scheme should be sufficient
	u, _ := url.Parse(strings.TrimPrefix(in, prefix))
	debugLog("repaired URI: original: %q, repaired: %q", in, u.String())

	return u, u.String()
}

// fixWindowsURI tolerates an absolute file path on windows such as C:\Base\File.yaml or \\host\share\Base\File.yaml
// and makes it a canonical URI: file:///c:/base/file.yaml
//
// Catch 22 notes for Windows:
//
// * There may be a drive letter on windows (it is lower-cased)
// * There may be a share UNC, e.g. \\server\folder\data.xml
// * Paths are case insensitive
// * Paths may already contain slashes
// * Paths must be slashed
//
// NOTE: there is no escaping. "/" may be valid separators just like "\".
// We don't use ToSlash() (which escapes everything) because windows now also
// tolerates the use of "/". Hence, both C:\File.yaml and C:/File.yaml will work.
func fixWindowsURI(u *url.URL, in string) {
	drive := filepath.VolumeName(in)

	if len(drive) > 0 {
		if len(u.Scheme) == 1 && strings.EqualFold(u.Scheme, drive[:1]) { // a path with a drive letter
			u.Scheme = fileScheme
			u.Host = ""
			u.Path = strings.Join([]string{drive, u.Opaque, u.Path}, `/`) // reconstruct the full path component (no fragment, no query)
		} else if u.Host == "" && strings.HasPrefix(u.Path, drive) { // a path with a \\host volume
			// NOTE: the special host@port syntax for UNC is not supported (yet)
			u.Scheme = fileScheme

			// this is a modified version of filepath.Dir() to apply on the VolumeName itself
			i := len(drive) - 1
			for i >= 0 && !os.IsPathSeparator(drive[i]) {
				i--
			}
			host := drive[:i] // \\host\share => host

			u.Path = strings.TrimPrefix(u.Path, host)
			u.Host = strings.TrimPrefix(host, `\\`)
		}

		u.Opaque = ""
		u.Path = strings.ReplaceAll(strings.ToLower(u.Path), `\`, `/`)

		// ensure we form an absolute path
		if !strings.HasPrefix(u.Path, "/") {
			u.Path = "/" + u.Path
		}

		u.Path = path.Clean(u.Path)

		return
	}

	if u.Scheme == fileScheme {
		// Handle dodgy cases for file://{...} URIs on windows.
		// A canonical URI should always be followed by an absolute path.
		//
		// Examples:
		//   * file:///folder/file => valid, unchanged
		//   * file:///c:\folder\file => slashed
		//   * file:///./folder/file => valid, cleaned to remove the dot
		//   * file:///.\folder\file => remapped to cwd
		//   * file:///. => dodgy, remapped to / (consistent with the behavior on unix)
		//   * file:///.. => dodgy, remapped to / (consistent with the behavior on unix)
		if (!path.IsAbs(u.Path) && !filepath.IsAbs(u.Path)) || (strings.HasPrefix(u.Path, `/.`) && strings.Contains(u.Path, `\`)) {
			// ensure we form an absolute path
			u.Path, _ = filepath.Abs(strings.TrimLeft(u.Path, `/`))
			if !strings.HasPrefix(u.Path, "/") {
				u.Path = "/" + u.Path
			}
		}
		u.Path = strings.ToLower(u.Path)
	}

	// NOTE: lower case normalization does not propagate to inner resources,
	// generated when rebasing: when joining a relative URI with a file to an absolute base,
	// only the base is currently lower-cased.
	//
	// For now, we assume this is good enough for most use cases
	// and try not to generate too many differences
	// between the output produced on different platforms.
	u.Path = path.Clean(strings.ReplaceAll(u.Path, `\`, `/`))
}

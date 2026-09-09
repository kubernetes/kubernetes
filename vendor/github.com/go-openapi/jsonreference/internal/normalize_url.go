// SPDX-FileCopyrightText: Copyright (c) 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"net/url"
	"regexp"
	"strings"
)

const (
	defaultHTTPPort  = ":80"
	defaultHTTPSPort = ":443"
)

// Regular expressions used by the normalizations.
var (
	rxPort       = regexp.MustCompile(`(:\d+)/?$`)
	rxDupSlashes = regexp.MustCompile(`/{2,}`)
)

// NormalizeURL will normalize the specified URL
// This was added to replace a previous call to the no longer maintained purell library:
// The call that was used looked like the following:
//
//	url.Parse(purell.NormalizeURL(parsed, purell.FlagsSafe|purell.FlagRemoveDuplicateSlashes))
//
// To explain all that was included in the call above, purell.FlagsSafe was really just the following:
//   - FlagLowercaseScheme
//   - FlagLowercaseHost
//   - FlagRemoveDefaultPort
//   - FlagRemoveDuplicateSlashes (and this was mixed in with the |)
//
// This also normalizes the URL into its urlencoded form by removing RawPath and RawFragment.
func NormalizeURL(u *url.URL) {
	lowercaseScheme(u)
	lowercaseHost(u)
	removeDefaultPort(u)
	removeDuplicateSlashes(u)

	u.RawPath = ""
	u.RawFragment = ""
}

func lowercaseScheme(u *url.URL) {
	if len(u.Scheme) > 0 {
		u.Scheme = strings.ToLower(u.Scheme)
	}
}

func lowercaseHost(u *url.URL) {
	if len(u.Host) > 0 {
		u.Host = strings.ToLower(u.Host)
	}
}

func removeDefaultPort(u *url.URL) {
	if len(u.Host) > 0 {
		scheme := strings.ToLower(u.Scheme)
		u.Host = rxPort.ReplaceAllStringFunc(u.Host, func(val string) string {
			if (scheme == "http" && val == defaultHTTPPort) || (scheme == "https" && val == defaultHTTPSPort) {
				return ""
			}
			return val
		})
	}
}

func removeDuplicateSlashes(u *url.URL) {
	if len(u.Path) > 0 {
		u.Path = rxDupSlashes.ReplaceAllString(u.Path, "/")
	}
}

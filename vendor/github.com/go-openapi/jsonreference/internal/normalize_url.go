// SPDX-FileCopyrightText: Copyright (c) 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package internal

import (
	"net/url"
	"strings"
)

const (
	defaultHTTPPort  = "80"
	defaultHTTPSPort = "443"
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

// removeDefaultPort drops :80 from an http URL and :443 from an https one.
//
// The port stays when dropping it would leave an authority url.Parse no longer accepts, so the
// shortened host is parsed before being kept. url.Parse reads "https://:a:443" as the host ":a"
// on port 443, and ":a" on its own is an invalid port, so "https://:a" no longer parses.
//
// A degenerate authority can spell a default port twice - url.Parse reads "http://:80:80" as the
// host ":80" on port 80 - so removal repeats until nothing more comes off. Each pass shortens the
// host, so the loop ends, and normalizing the result again changes nothing.
func removeDefaultPort(u *url.URL) {
	for {
		port := u.Port()
		if port == "" || port != defaultPortForScheme(strings.ToLower(u.Scheme)) {
			return
		}

		host := strings.TrimSuffix(u.Host, ":"+port)
		if _, err := url.Parse("//" + host); err != nil {
			return
		}

		u.Host = host
	}
}

func defaultPortForScheme(scheme string) string {
	switch scheme {
	case "http":
		return defaultHTTPPort
	case "https":
		return defaultHTTPSPort
	default:
		return ""
	}
}

// removeDuplicateSlashes collapses every run of slashes in the path to a single one, however
// long the run is: "/a//b///c" becomes "/a/b/c".
//
// A path holding no "//" is left as it is, which is the common case and costs one scan and no
// allocation.
func removeDuplicateSlashes(u *url.URL) {
	const doubleSlash = "//"

	start := strings.Index(u.Path, doubleSlash)
	if start < 0 {
		return
	}

	var collapsed strings.Builder
	collapsed.Grow(len(u.Path))
	collapsed.WriteString(u.Path[:start+1]) // everything up to and including the first slash of the run

	for i := start + 1; i < len(u.Path); i++ {
		c := u.Path[i]
		if c == '/' && u.Path[i-1] == '/' {
			continue
		}
		collapsed.WriteByte(c)
	}

	u.Path = collapsed.String()
}

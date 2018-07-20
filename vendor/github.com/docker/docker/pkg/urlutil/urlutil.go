// Package urlutil provides helper function to check urls kind.
// It supports http urls, git urls and transport url (tcp://, …)
package urlutil

import (
	"regexp"
	"strings"
)

var (
	validPrefixes = map[string][]string{
		"url":       {"http://", "https://"},
		"git":       {"git://", "github.com/", "git@"},
		"transport": {"tcp://", "tcp+tls://", "udp://", "unix://", "unixgram://"},
	}
	urlPathWithFragmentSuffix = regexp.MustCompile(".git(?:#.+)?$")
)

// IsURL returns true if the provided str is an HTTP(S) URL.
func IsURL(str string) bool {
	return checkURL(str, "url")
}

// IsGitURL returns true if the provided str is a git repository URL.
func IsGitURL(str string) bool {
	if IsURL(str) && urlPathWithFragmentSuffix.MatchString(str) {
		return true
	}
	return checkURL(str, "git")
}

// IsTransportURL returns true if the provided str is a transport (tcp, tcp+tls, udp, unix) URL.
func IsTransportURL(str string) bool {
	return checkURL(str, "transport")
}

func checkURL(str, kind string) bool {
	for _, prefix := range validPrefixes[kind] {
		if strings.HasPrefix(str, prefix) {
			return true
		}
	}
	return false
}

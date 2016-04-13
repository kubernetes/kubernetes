package urlutil

import (
	"regexp"
	"strings"
)

var (
	validPrefixes = map[string][]string{
		"url":       {"http://", "https://"},
		"git":       {"git://", "github.com/", "git@"},
		"transport": {"tcp://", "udp://", "unix://"},
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

// IsGitTransport returns true if the provided str is a git transport by inspecting
// the prefix of the string for known protocols used in git.
func IsGitTransport(str string) bool {
	return IsURL(str) || strings.HasPrefix(str, "git://") || strings.HasPrefix(str, "git@")
}

// IsTransportURL returns true if the provided str is a transport (tcp, udp, unix) URL.
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

/*
Package purell offers URL normalization as described on the wikipedia page:
http://en.wikipedia.org/wiki/URL_normalization
*/
package purell

import (
	"bytes"
	"fmt"
	"net/url"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/PuerkitoBio/urlesc"
	"golang.org/x/net/idna"
	"golang.org/x/text/unicode/norm"
	"golang.org/x/text/width"
)

// A set of normalization flags determines how a URL will
// be normalized.
type NormalizationFlags uint

const (
	// Safe normalizations
	FlagLowercaseScheme           NormalizationFlags = 1 << iota // HTTP://host -> http://host, applied by default in Go1.1
	FlagLowercaseHost                                            // http://HOST -> http://host
	FlagUppercaseEscapes                                         // http://host/t%ef -> http://host/t%EF
	FlagDecodeUnnecessaryEscapes                                 // http://host/t%41 -> http://host/tA
	FlagEncodeNecessaryEscapes                                   // http://host/!"#$ -> http://host/%21%22#$
	FlagRemoveDefaultPort                                        // http://host:80 -> http://host
	FlagRemoveEmptyQuerySeparator                                // http://host/path? -> http://host/path

	// Usually safe normalizations
	FlagRemoveTrailingSlash // http://host/path/ -> http://host/path
	FlagAddTrailingSlash    // http://host/path -> http://host/path/ (should choose only one of these add/remove trailing slash flags)
	FlagRemoveDotSegments   // http://host/path/./a/b/../c -> http://host/path/a/c

	// Unsafe normalizations
	FlagRemoveDirectoryIndex   // http://host/path/index.html -> http://host/path/
	FlagRemoveFragment         // http://host/path#fragment -> http://host/path
	FlagForceHTTP              // https://host -> http://host
	FlagRemoveDuplicateSlashes // http://host/path//a///b -> http://host/path/a/b
	FlagRemoveWWW              // http://www.host/ -> http://host/
	FlagAddWWW                 // http://host/ -> http://www.host/ (should choose only one of these add/remove WWW flags)
	FlagSortQuery              // http://host/path?c=3&b=2&a=1&b=1 -> http://host/path?a=1&b=1&b=2&c=3

	// Normalizations not in the wikipedia article, required to cover tests cases
	// submitted by jehiah
	FlagDecodeDWORDHost           // http://1113982867 -> http://66.102.7.147
	FlagDecodeOctalHost           // http://0102.0146.07.0223 -> http://66.102.7.147
	FlagDecodeHexHost             // http://0x42660793 -> http://66.102.7.147
	FlagRemoveUnnecessaryHostDots // http://.host../path -> http://host/path
	FlagRemoveEmptyPortSeparator  // http://host:/path -> http://host/path

	// Convenience set of safe normalizations
	FlagsSafe NormalizationFlags = FlagLowercaseHost | FlagLowercaseScheme | FlagUppercaseEscapes | FlagDecodeUnnecessaryEscapes | FlagEncodeNecessaryEscapes | FlagRemoveDefaultPort | FlagRemoveEmptyQuerySeparator

	// For convenience sets, "greedy" uses the "remove trailing slash" and "remove www. prefix" flags,
	// while "non-greedy" uses the "add (or keep) the trailing slash" and "add www. prefix".

	// Convenience set of usually safe normalizations (includes FlagsSafe)
	FlagsUsuallySafeGreedy    NormalizationFlags = FlagsSafe | FlagRemoveTrailingSlash | FlagRemoveDotSegments
	FlagsUsuallySafeNonGreedy NormalizationFlags = FlagsSafe | FlagAddTrailingSlash | FlagRemoveDotSegments

	// Convenience set of unsafe normalizations (includes FlagsUsuallySafe)
	FlagsUnsafeGreedy    NormalizationFlags = FlagsUsuallySafeGreedy | FlagRemoveDirectoryIndex | FlagRemoveFragment | FlagForceHTTP | FlagRemoveDuplicateSlashes | FlagRemoveWWW | FlagSortQuery
	FlagsUnsafeNonGreedy NormalizationFlags = FlagsUsuallySafeNonGreedy | FlagRemoveDirectoryIndex | FlagRemoveFragment | FlagForceHTTP | FlagRemoveDuplicateSlashes | FlagAddWWW | FlagSortQuery

	// Convenience set of all available flags
	FlagsAllGreedy    = FlagsUnsafeGreedy | FlagDecodeDWORDHost | FlagDecodeOctalHost | FlagDecodeHexHost | FlagRemoveUnnecessaryHostDots | FlagRemoveEmptyPortSeparator
	FlagsAllNonGreedy = FlagsUnsafeNonGreedy | FlagDecodeDWORDHost | FlagDecodeOctalHost | FlagDecodeHexHost | FlagRemoveUnnecessaryHostDots | FlagRemoveEmptyPortSeparator
)

const (
	defaultHttpPort  = ":80"
	defaultHttpsPort = ":443"
)

// Regular expressions used by the normalizations
var rxPort = regexp.MustCompile(`(:\d+)/?$`)
var rxDirIndex = regexp.MustCompile(`(^|/)((?:default|index)\.\w{1,4})$`)
var rxDupSlashes = regexp.MustCompile(`/{2,}`)
var rxDWORDHost = regexp.MustCompile(`^(\d+)((?:\.+)?(?:\:\d*)?)$`)
var rxOctalHost = regexp.MustCompile(`^(0\d*)\.(0\d*)\.(0\d*)\.(0\d*)((?:\.+)?(?:\:\d*)?)$`)
var rxHexHost = regexp.MustCompile(`^0x([0-9A-Fa-f]+)((?:\.+)?(?:\:\d*)?)$`)
var rxHostDots = regexp.MustCompile(`^(.+?)(:\d+)?$`)
var rxEmptyPort = regexp.MustCompile(`:+$`)

// Map of flags to implementation function.
// FlagDecodeUnnecessaryEscapes has no action, since it is done automatically
// by parsing the string as an URL. Same for FlagUppercaseEscapes and FlagRemoveEmptyQuerySeparator.

// Since maps have undefined traversing order, make a slice of ordered keys
var flagsOrder = []NormalizationFlags{
	FlagLowercaseScheme,
	FlagLowercaseHost,
	FlagRemoveDefaultPort,
	FlagRemoveDirectoryIndex,
	FlagRemoveDotSegments,
	FlagRemoveFragment,
	FlagForceHTTP, // Must be after remove default port (because https=443/http=80)
	FlagRemoveDuplicateSlashes,
	FlagRemoveWWW,
	FlagAddWWW,
	FlagSortQuery,
	FlagDecodeDWORDHost,
	FlagDecodeOctalHost,
	FlagDecodeHexHost,
	FlagRemoveUnnecessaryHostDots,
	FlagRemoveEmptyPortSeparator,
	FlagRemoveTrailingSlash, // These two (add/remove trailing slash) must be last
	FlagAddTrailingSlash,
}

// ... and then the map, where order is unimportant
var flags = map[NormalizationFlags]func(*url.URL){
	FlagLowercaseScheme:           lowercaseScheme,
	FlagLowercaseHost:             lowercaseHost,
	FlagRemoveDefaultPort:         removeDefaultPort,
	FlagRemoveDirectoryIndex:      removeDirectoryIndex,
	FlagRemoveDotSegments:         removeDotSegments,
	FlagRemoveFragment:            removeFragment,
	FlagForceHTTP:                 forceHTTP,
	FlagRemoveDuplicateSlashes:    removeDuplicateSlashes,
	FlagRemoveWWW:                 removeWWW,
	FlagAddWWW:                    addWWW,
	FlagSortQuery:                 sortQuery,
	FlagDecodeDWORDHost:           decodeDWORDHost,
	FlagDecodeOctalHost:           decodeOctalHost,
	FlagDecodeHexHost:             decodeHexHost,
	FlagRemoveUnnecessaryHostDots: removeUnncessaryHostDots,
	FlagRemoveEmptyPortSeparator:  removeEmptyPortSeparator,
	FlagRemoveTrailingSlash:       removeTrailingSlash,
	FlagAddTrailingSlash:          addTrailingSlash,
}

// MustNormalizeURLString returns the normalized string, and panics if an error occurs.
// It takes an URL string as input, as well as the normalization flags.
func MustNormalizeURLString(u string, f NormalizationFlags) string {
	result, e := NormalizeURLString(u, f)
	if e != nil {
		panic(e)
	}
	return result
}

// NormalizeURLString returns the normalized string, or an error if it can't be parsed into an URL object.
// It takes an URL string as input, as well as the normalization flags.
func NormalizeURLString(u string, f NormalizationFlags) (string, error) {
	parsed, err := url.Parse(u)
	if err != nil {
		return "", err
	}

	if f&FlagLowercaseHost == FlagLowercaseHost {
		parsed.Host = strings.ToLower(parsed.Host)
	}

	// The idna package doesn't fully conform to RFC 5895
	// (https://tools.ietf.org/html/rfc5895), so we do it here.
	// Taken from Go 1.8 cycle source, courtesy of bradfitz.
	// TODO: Remove when (if?) idna package conforms to RFC 5895.
	parsed.Host = width.Fold.String(parsed.Host)
	parsed.Host = norm.NFC.String(parsed.Host)
	if parsed.Host, err = idna.ToASCII(parsed.Host); err != nil {
		return "", err
	}

	return NormalizeURL(parsed, f), nil
}

// NormalizeURL returns the normalized string.
// It takes a parsed URL object as input, as well as the normalization flags.
func NormalizeURL(u *url.URL, f NormalizationFlags) string {
	for _, k := range flagsOrder {
		if f&k == k {
			flags[k](u)
		}
	}
	return urlesc.Escape(u)
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
			if (scheme == "http" && val == defaultHttpPort) || (scheme == "https" && val == defaultHttpsPort) {
				return ""
			}
			return val
		})
	}
}

func removeTrailingSlash(u *url.URL) {
	if l := len(u.Path); l > 0 {
		if strings.HasSuffix(u.Path, "/") {
			u.Path = u.Path[:l-1]
		}
	} else if l = len(u.Host); l > 0 {
		if strings.HasSuffix(u.Host, "/") {
			u.Host = u.Host[:l-1]
		}
	}
}

func addTrailingSlash(u *url.URL) {
	if l := len(u.Path); l > 0 {
		if !strings.HasSuffix(u.Path, "/") {
			u.Path += "/"
		}
	} else if l = len(u.Host); l > 0 {
		if !strings.HasSuffix(u.Host, "/") {
			u.Host += "/"
		}
	}
}

func removeDotSegments(u *url.URL) {
	if len(u.Path) > 0 {
		var dotFree []string
		var lastIsDot bool

		sections := strings.Split(u.Path, "/")
		for _, s := range sections {
			if s == ".." {
				if len(dotFree) > 0 {
					dotFree = dotFree[:len(dotFree)-1]
				}
			} else if s != "." {
				dotFree = append(dotFree, s)
			}
			lastIsDot = (s == "." || s == "..")
		}
		// Special case if host does not end with / and new path does not begin with /
		u.Path = strings.Join(dotFree, "/")
		if u.Host != "" && !strings.HasSuffix(u.Host, "/") && !strings.HasPrefix(u.Path, "/") {
			u.Path = "/" + u.Path
		}
		// Special case if the last segment was a dot, make sure the path ends with a slash
		if lastIsDot && !strings.HasSuffix(u.Path, "/") {
			u.Path += "/"
		}
	}
}

func removeDirectoryIndex(u *url.URL) {
	if len(u.Path) > 0 {
		u.Path = rxDirIndex.ReplaceAllString(u.Path, "$1")
	}
}

func removeFragment(u *url.URL) {
	u.Fragment = ""
}

func forceHTTP(u *url.URL) {
	if strings.ToLower(u.Scheme) == "https" {
		u.Scheme = "http"
	}
}

func removeDuplicateSlashes(u *url.URL) {
	if len(u.Path) > 0 {
		u.Path = rxDupSlashes.ReplaceAllString(u.Path, "/")
	}
}

func removeWWW(u *url.URL) {
	if len(u.Host) > 0 && strings.HasPrefix(strings.ToLower(u.Host), "www.") {
		u.Host = u.Host[4:]
	}
}

func addWWW(u *url.URL) {
	if len(u.Host) > 0 && !strings.HasPrefix(strings.ToLower(u.Host), "www.") {
		u.Host = "www." + u.Host
	}
}

func sortQuery(u *url.URL) {
	q := u.Query()

	if len(q) > 0 {
		arKeys := make([]string, len(q))
		i := 0
		for k, _ := range q {
			arKeys[i] = k
			i++
		}
		sort.Strings(arKeys)
		buf := new(bytes.Buffer)
		for _, k := range arKeys {
			sort.Strings(q[k])
			for _, v := range q[k] {
				if buf.Len() > 0 {
					buf.WriteRune('&')
				}
				buf.WriteString(fmt.Sprintf("%s=%s", k, urlesc.QueryEscape(v)))
			}
		}

		// Rebuild the raw query string
		u.RawQuery = buf.String()
	}
}

func decodeDWORDHost(u *url.URL) {
	if len(u.Host) > 0 {
		if matches := rxDWORDHost.FindStringSubmatch(u.Host); len(matches) > 2 {
			var parts [4]int64

			dword, _ := strconv.ParseInt(matches[1], 10, 0)
			for i, shift := range []uint{24, 16, 8, 0} {
				parts[i] = dword >> shift & 0xFF
			}
			u.Host = fmt.Sprintf("%d.%d.%d.%d%s", parts[0], parts[1], parts[2], parts[3], matches[2])
		}
	}
}

func decodeOctalHost(u *url.URL) {
	if len(u.Host) > 0 {
		if matches := rxOctalHost.FindStringSubmatch(u.Host); len(matches) > 5 {
			var parts [4]int64

			for i := 1; i <= 4; i++ {
				parts[i-1], _ = strconv.ParseInt(matches[i], 8, 0)
			}
			u.Host = fmt.Sprintf("%d.%d.%d.%d%s", parts[0], parts[1], parts[2], parts[3], matches[5])
		}
	}
}

func decodeHexHost(u *url.URL) {
	if len(u.Host) > 0 {
		if matches := rxHexHost.FindStringSubmatch(u.Host); len(matches) > 2 {
			// Conversion is safe because of regex validation
			parsed, _ := strconv.ParseInt(matches[1], 16, 0)
			// Set host as DWORD (base 10) encoded host
			u.Host = fmt.Sprintf("%d%s", parsed, matches[2])
			// The rest is the same as decoding a DWORD host
			decodeDWORDHost(u)
		}
	}
}

func removeUnncessaryHostDots(u *url.URL) {
	if len(u.Host) > 0 {
		if matches := rxHostDots.FindStringSubmatch(u.Host); len(matches) > 1 {
			// Trim the leading and trailing dots
			u.Host = strings.Trim(matches[1], ".")
			if len(matches) > 2 {
				u.Host += matches[2]
			}
		}
	}
}

func removeEmptyPortSeparator(u *url.URL) {
	if len(u.Host) > 0 {
		u.Host = rxEmptyPort.ReplaceAllString(u.Host, "")
	}
}

package url

import (
	"fmt"
	"net/url"
	"path/filepath"
	"strings"
)

func parse(rawURL string) (*url.URL, error) {
	// Make sure we're using "/" since URLs are "/"-based.
	rawURL = filepath.ToSlash(rawURL)

	u, err := url.Parse(rawURL)
	if err != nil {
		return nil, err
	}

	if len(rawURL) > 1 && rawURL[1] == ':' {
		// Assume we're dealing with a drive letter file path where the drive
		// letter has been parsed into the URL Scheme, and the rest of the path
		// has been parsed into the URL Path without the leading ':' character.
		u.Path = fmt.Sprintf("%s:%s", string(rawURL[0]), u.Path)
		u.Scheme = ""
	}

	if len(u.Host) > 1 && u.Host[1] == ':' && strings.HasPrefix(rawURL, "file://") {
		// Assume we're dealing with a drive letter file path where the drive
		// letter has been parsed into the URL Host.
		u.Path = fmt.Sprintf("%s%s", u.Host, u.Path)
		u.Host = ""
	}

	// Remove leading slash for absolute file paths.
	if len(u.Path) > 2 && u.Path[0] == '/' && u.Path[2] == ':' {
		u.Path = u.Path[1:]
	}

	return u, err
}

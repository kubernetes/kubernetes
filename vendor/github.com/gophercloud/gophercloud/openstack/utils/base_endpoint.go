package utils

import (
	"net/url"
	"regexp"
	"strings"
)

// BaseEndpoint will return a URL without the /vX.Y
// portion of the URL.
func BaseEndpoint(endpoint string) (string, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return "", err
	}

	u.RawQuery, u.Fragment = "", ""

	path := u.Path
	versionRe := regexp.MustCompile("v[0-9.]+/?")

	if version := versionRe.FindString(path); version != "" {
		versionIndex := strings.Index(path, version)
		u.Path = path[:versionIndex]
	}

	return u.String(), nil
}

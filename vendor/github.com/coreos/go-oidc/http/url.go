package http

import (
	"errors"
	"net/url"
)

// ParseNonEmptyURL checks that a string is a parsable URL which is also not empty
// since `url.Parse("")` does not return an error. Must contian a scheme and a host.
func ParseNonEmptyURL(u string) (*url.URL, error) {
	if u == "" {
		return nil, errors.New("url is empty")
	}

	ur, err := url.Parse(u)
	if err != nil {
		return nil, err
	}

	if ur.Scheme == "" {
		return nil, errors.New("url scheme is empty")
	}

	if ur.Host == "" {
		return nil, errors.New("url host is empty")
	}

	return ur, nil
}

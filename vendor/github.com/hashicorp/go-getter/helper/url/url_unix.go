// +build !windows

package url

import (
	"net/url"
)

func parse(rawURL string) (*url.URL, error) {
	return url.Parse(rawURL)
}

// +build go1.8

package httptreemux

import "net/url"

func unescape(path string) (string, error) {
	return url.PathUnescape(path)
}

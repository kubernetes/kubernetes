// +build go1.8

package swag

import "net/url"

func pathUnescape(path string) (string, error) {
	return url.PathUnescape(path)
}

package netutil

import (
	"net/url"
)

// MergeQuery appends additional query values to an existing URL.
func MergeQuery(u url.URL, q url.Values) url.URL {
	uv := u.Query()
	for k, vs := range q {
		for _, v := range vs {
			uv.Add(k, v)
		}
	}
	u.RawQuery = uv.Encode()
	return u
}

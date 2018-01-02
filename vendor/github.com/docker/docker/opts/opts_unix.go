// +build !windows

package opts

// DefaultHTTPHost Default HTTP Host used if only port is provided to -H flag e.g. dockerd -H tcp://:8080
const DefaultHTTPHost = "localhost"

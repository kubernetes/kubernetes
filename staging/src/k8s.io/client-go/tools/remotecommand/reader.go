/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package remotecommand

import (
	"io"
)

// readerWrapper delegates to an io.Reader so that only the io.Reader interface is implemented,
// to keep io.Copy from doing things we don't want when copying from the reader to the data stream.
//
// If the Stdin io.Reader provided to remotecommand implements a WriteTo function (like bytes.Buffer does[1]),
// io.Copy calls that method[2] to attempt to write the entire buffer to the stream in one call.
// That results in an oversized call to spdystream.Stream#Write [3],
// which results in a single oversized data frame[4] that is too large.
//
// [1] https://golang.org/pkg/bytes/#Buffer.WriteTo
// [2] https://golang.org/pkg/io/#Copy
// [3] https://github.com/kubernetes/kubernetes/blob/90295640ef87db9daa0144c5617afe889e7992b2/vendor/github.com/docker/spdystream/stream.go#L66-L73
// [4] https://github.com/kubernetes/kubernetes/blob/90295640ef87db9daa0144c5617afe889e7992b2/vendor/github.com/docker/spdystream/spdy/write.go#L302-L304
type readerWrapper struct {
	reader io.Reader
}

func (r readerWrapper) Read(p []byte) (int, error) {
	return r.reader.Read(p)
}

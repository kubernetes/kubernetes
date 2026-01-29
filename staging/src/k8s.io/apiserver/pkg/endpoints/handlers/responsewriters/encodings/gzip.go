/*
Copyright 2025 The Kubernetes Authors.

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

package encodings

import (
	"compress/gzip"
	"io"
	"sync"
)

var Gzip Interface = gzipEncoding{}

type gzipEncoding struct{}

func (gzipEncoding) NewWriter(w io.Writer) io.WriteCloser {
	encoder := gzipPool.Get().(*gzip.Writer)
	encoder.Reset(w)
	return &gzipEncoder{Writer: encoder}
}

func (gzipEncoding) EncoderName() string {
	return "gzip"
}

type gzipEncoder struct {
	*gzip.Writer
}

func (e *gzipEncoder) Close() error {
	err := e.Writer.Close()
	e.Writer.Reset(nil)
	gzipPool.Put(e.Writer)
	return err
}

var gzipPool = &sync.Pool{
	New: func() interface{} {
		gw, err := gzip.NewWriterLevel(nil, defaultGzipContentEncodingLevel)
		if err != nil {
			panic(err)
		}
		return gw
	},
}

const (
	// defaultGzipContentEncodingLevel is set to 1 which uses least CPU compared to higher levels, yet offers
	// similar compression ratios (off by at most 1.5x, but typically within 1.1x-1.3x). For further details see -
	// https://github.com/kubernetes/kubernetes/issues/112296
	defaultGzipContentEncodingLevel = 1
)

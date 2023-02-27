// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

package comm

import (
	"compress/gzip"
	"io"
)

func gzipDecompress(r io.Reader) io.Reader {
	gzipReader, _ := gzip.NewReader(r)

	pipeOut, pipeIn := io.Pipe()
	go func() {
		// decompression bomb would have to come from Azure services.
		// If we want to limit, we should do that in comm.do().
		_, err := io.Copy(pipeIn, gzipReader) //nolint
		if err != nil {
			// don't need the error.
			pipeIn.CloseWithError(err) //nolint
			gzipReader.Close()
			return
		}
		if err := gzipReader.Close(); err != nil {
			// don't need the error.
			pipeIn.CloseWithError(err) //nolint
			return
		}
		pipeIn.Close()
	}()
	return pipeOut
}

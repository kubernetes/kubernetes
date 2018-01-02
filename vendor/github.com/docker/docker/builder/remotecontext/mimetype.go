package remotecontext

import (
	"mime"
	"net/http"
)

// mimeTypes stores the MIME content type.
var mimeTypes = struct {
	TextPlain   string
	OctetStream string
}{"text/plain", "application/octet-stream"}

// detectContentType returns a best guess representation of the MIME
// content type for the bytes at c.  The value detected by
// http.DetectContentType is guaranteed not be nil, defaulting to
// application/octet-stream when a better guess cannot be made. The
// result of this detection is then run through mime.ParseMediaType()
// which separates the actual MIME string from any parameters.
func detectContentType(c []byte) (string, map[string]string, error) {
	ct := http.DetectContentType(c)
	contentType, args, err := mime.ParseMediaType(ct)
	if err != nil {
		return "", nil, err
	}
	return contentType, args, nil
}

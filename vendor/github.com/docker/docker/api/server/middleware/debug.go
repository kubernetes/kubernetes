package middleware

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"strings"

	"github.com/docker/docker/api/server/httputils"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

// DebugRequestMiddleware dumps the request to logger
func DebugRequestMiddleware(handler func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error) func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	return func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
		logrus.Debugf("Calling %s %s", r.Method, r.RequestURI)

		if r.Method != "POST" {
			return handler(ctx, w, r, vars)
		}
		if err := httputils.CheckForJSON(r); err != nil {
			return handler(ctx, w, r, vars)
		}
		maxBodySize := 4096 // 4KB
		if r.ContentLength > int64(maxBodySize) {
			return handler(ctx, w, r, vars)
		}

		body := r.Body
		bufReader := bufio.NewReaderSize(body, maxBodySize)
		r.Body = ioutils.NewReadCloserWrapper(bufReader, func() error { return body.Close() })

		b, err := bufReader.Peek(maxBodySize)
		if err != io.EOF {
			// either there was an error reading, or the buffer is full (in which case the request is too large)
			return handler(ctx, w, r, vars)
		}

		var postForm map[string]interface{}
		if err := json.Unmarshal(b, &postForm); err == nil {
			maskSecretKeys(postForm, r.RequestURI)
			formStr, errMarshal := json.Marshal(postForm)
			if errMarshal == nil {
				logrus.Debugf("form data: %s", string(formStr))
			} else {
				logrus.Debugf("form data: %q", postForm)
			}
		}

		return handler(ctx, w, r, vars)
	}
}

func maskSecretKeys(inp interface{}, path string) {
	// Remove any query string from the path
	idx := strings.Index(path, "?")
	if idx != -1 {
		path = path[:idx]
	}
	// Remove trailing / characters
	path = strings.TrimRight(path, "/")

	if arr, ok := inp.([]interface{}); ok {
		for _, f := range arr {
			maskSecretKeys(f, path)
		}
		return
	}

	if form, ok := inp.(map[string]interface{}); ok {
	loop0:
		for k, v := range form {
			for _, m := range []string{"password", "secret", "jointoken", "unlockkey", "signingcakey"} {
				if strings.EqualFold(m, k) {
					form[k] = "*****"
					continue loop0
				}
			}
			maskSecretKeys(v, path)
		}

		// Route-specific redactions
		if strings.HasSuffix(path, "/secrets/create") {
			for k := range form {
				if k == "Data" {
					form[k] = "*****"
				}
			}
		}
	}
}

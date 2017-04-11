package handlers

import (
	"errors"
	"io"
	"net/http"

	ctxu "github.com/docker/distribution/context"
	"github.com/docker/distribution/registry/api/errcode"
)

// closeResources closes all the provided resources after running the target
// handler.
func closeResources(handler http.Handler, closers ...io.Closer) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		for _, closer := range closers {
			defer closer.Close()
		}
		handler.ServeHTTP(w, r)
	})
}

// copyFullPayload copies the payload of a HTTP request to destWriter. If it
// receives less content than expected, and the client disconnected during the
// upload, it avoids sending a 400 error to keep the logs cleaner.
func copyFullPayload(responseWriter http.ResponseWriter, r *http.Request, destWriter io.Writer, context ctxu.Context, action string, errSlice *errcode.Errors) error {
	// Get a channel that tells us if the client disconnects
	var clientClosed <-chan bool
	if notifier, ok := responseWriter.(http.CloseNotifier); ok {
		clientClosed = notifier.CloseNotify()
	} else {
		ctxu.GetLogger(context).Warnf("the ResponseWriter does not implement CloseNotifier (type: %T)", responseWriter)
	}

	// Read in the data, if any.
	copied, err := io.Copy(destWriter, r.Body)
	if clientClosed != nil && (err != nil || (r.ContentLength > 0 && copied < r.ContentLength)) {
		// Didn't receive as much content as expected. Did the client
		// disconnect during the request? If so, avoid returning a 400
		// error to keep the logs cleaner.
		select {
		case <-clientClosed:
			// Set the response code to "499 Client Closed Request"
			// Even though the connection has already been closed,
			// this causes the logger to pick up a 499 error
			// instead of showing 0 for the HTTP status.
			responseWriter.WriteHeader(499)

			ctxu.GetLoggerWithFields(context, map[interface{}]interface{}{
				"error":         err,
				"copied":        copied,
				"contentLength": r.ContentLength,
			}, "error", "copied", "contentLength").Error("client disconnected during " + action)
			return errors.New("client disconnected")
		default:
		}
	}

	if err != nil {
		ctxu.GetLogger(context).Errorf("unknown error reading request payload: %v", err)
		*errSlice = append(*errSlice, errcode.ErrorCodeUnknown.WithDetail(err))
		return err
	}

	return nil
}

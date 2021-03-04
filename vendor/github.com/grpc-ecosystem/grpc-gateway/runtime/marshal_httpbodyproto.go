package runtime

import (
	"google.golang.org/genproto/googleapis/api/httpbody"
)

// SetHTTPBodyMarshaler overwrite the default marshaler with the HTTPBodyMarshaler
func SetHTTPBodyMarshaler(serveMux *ServeMux) {
	serveMux.marshalers.mimeMap[MIMEWildcard] = &HTTPBodyMarshaler{
		Marshaler: &JSONPb{OrigName: true},
	}
}

// HTTPBodyMarshaler is a Marshaler which supports marshaling of a
// google.api.HttpBody message as the full response body if it is
// the actual message used as the response. If not, then this will
// simply fallback to the Marshaler specified as its default Marshaler.
type HTTPBodyMarshaler struct {
	Marshaler
}

// ContentType implementation to keep backwards compatability with marshal interface
func (h *HTTPBodyMarshaler) ContentType() string {
	return h.ContentTypeFromMessage(nil)
}

// ContentTypeFromMessage in case v is a google.api.HttpBody message it returns
// its specified content type otherwise fall back to the default Marshaler.
func (h *HTTPBodyMarshaler) ContentTypeFromMessage(v interface{}) string {
	if httpBody, ok := v.(*httpbody.HttpBody); ok {
		return httpBody.GetContentType()
	}
	return h.Marshaler.ContentType()
}

// Marshal marshals "v" by returning the body bytes if v is a
// google.api.HttpBody message, otherwise it falls back to the default Marshaler.
func (h *HTTPBodyMarshaler) Marshal(v interface{}) ([]byte, error) {
	if httpBody, ok := v.(*httpbody.HttpBody); ok {
		return httpBody.Data, nil
	}
	return h.Marshaler.Marshal(v)
}

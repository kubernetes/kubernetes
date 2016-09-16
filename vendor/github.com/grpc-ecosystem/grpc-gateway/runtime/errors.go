package runtime

import (
	"io"
	"net/http"

	"github.com/golang/protobuf/proto"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
)

// HTTPStatusFromCode converts a gRPC error code into the corresponding HTTP response status.
func HTTPStatusFromCode(code codes.Code) int {
	switch code {
	case codes.OK:
		return http.StatusOK
	case codes.Canceled:
		return http.StatusRequestTimeout
	case codes.Unknown:
		return http.StatusInternalServerError
	case codes.InvalidArgument:
		return http.StatusBadRequest
	case codes.DeadlineExceeded:
		return http.StatusRequestTimeout
	case codes.NotFound:
		return http.StatusNotFound
	case codes.AlreadyExists:
		return http.StatusConflict
	case codes.PermissionDenied:
		return http.StatusForbidden
	case codes.Unauthenticated:
		return http.StatusUnauthorized
	case codes.ResourceExhausted:
		return http.StatusForbidden
	case codes.FailedPrecondition:
		return http.StatusPreconditionFailed
	case codes.Aborted:
		return http.StatusConflict
	case codes.OutOfRange:
		return http.StatusBadRequest
	case codes.Unimplemented:
		return http.StatusNotImplemented
	case codes.Internal:
		return http.StatusInternalServerError
	case codes.Unavailable:
		return http.StatusServiceUnavailable
	case codes.DataLoss:
		return http.StatusInternalServerError
	}

	grpclog.Printf("Unknown gRPC error code: %v", code)
	return http.StatusInternalServerError
}

var (
	// HTTPError replies to the request with the error.
	// You can set a custom function to this variable to customize error format.
	HTTPError = DefaultHTTPError
	// OtherErrorHandler handles the following error used by the gateway: StatusMethodNotAllowed StatusNotFound and StatusBadRequest
	OtherErrorHandler = DefaultOtherErrorHandler
)

type errorBody struct {
	Error string `json:"error"`
	Code  int    `json:"code"`
}

//Make this also conform to proto.Message for builtin JSONPb Marshaler
func (e *errorBody) Reset()         { *e = errorBody{} }
func (e *errorBody) String() string { return proto.CompactTextString(e) }
func (*errorBody) ProtoMessage()    {}

// DefaultHTTPError is the default implementation of HTTPError.
// If "err" is an error from gRPC system, the function replies with the status code mapped by HTTPStatusFromCode.
// If otherwise, it replies with http.StatusInternalServerError.
//
// The response body returned by this function is a JSON object,
// which contains a member whose key is "error" and whose value is err.Error().
func DefaultHTTPError(ctx context.Context, marshaler Marshaler, w http.ResponseWriter, _ *http.Request, err error) {
	const fallback = `{"error": "failed to marshal error message"}`

	w.Header().Del("Trailer")
	w.Header().Set("Content-Type", marshaler.ContentType())
	body := &errorBody{
		Error: grpc.ErrorDesc(err),
		Code:  int(grpc.Code(err)),
	}

	buf, merr := marshaler.Marshal(body)
	if merr != nil {
		grpclog.Printf("Failed to marshal error message %q: %v", body, merr)
		w.WriteHeader(http.StatusInternalServerError)
		if _, err := io.WriteString(w, fallback); err != nil {
			grpclog.Printf("Failed to write response: %v", err)
		}
		return
	}

	md, ok := ServerMetadataFromContext(ctx)
	if !ok {
		grpclog.Printf("Failed to extract ServerMetadata from context")
	}

	handleForwardResponseServerMetadata(w, md)
	handleForwardResponseTrailerHeader(w, md)
	st := HTTPStatusFromCode(grpc.Code(err))
	w.WriteHeader(st)
	if _, err := w.Write(buf); err != nil {
		grpclog.Printf("Failed to write response: %v", err)
	}

	handleForwardResponseTrailer(w, md)
}

// DefaultOtherErrorHandler is the default implementation of OtherErrorHandler.
// It simply writes a string representation of the given error into "w".
func DefaultOtherErrorHandler(w http.ResponseWriter, _ *http.Request, msg string, code int) {
	http.Error(w, msg, code)
}

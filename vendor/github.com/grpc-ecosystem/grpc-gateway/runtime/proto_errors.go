package runtime

import (
	"context"
	"io"
	"net/http"

	"github.com/golang/protobuf/ptypes/any"
	"github.com/grpc-ecosystem/grpc-gateway/internal"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/grpclog"
	"google.golang.org/grpc/status"
)

// StreamErrorHandlerFunc accepts an error as a gRPC error generated via status package and translates it into a
// a proto struct used to represent error at the end of a stream.
type StreamErrorHandlerFunc func(context.Context, error) *StreamError

// StreamError is the payload for the final message in a server stream in the event that the server returns an
// error after a response message has already been sent.
type StreamError internal.StreamError

// ProtoErrorHandlerFunc handles the error as a gRPC error generated via status package and replies to the request.
type ProtoErrorHandlerFunc func(context.Context, *ServeMux, Marshaler, http.ResponseWriter, *http.Request, error)

var _ ProtoErrorHandlerFunc = DefaultHTTPProtoErrorHandler

// DefaultHTTPProtoErrorHandler is an implementation of HTTPError.
// If "err" is an error from gRPC system, the function replies with the status code mapped by HTTPStatusFromCode.
// If otherwise, it replies with http.StatusInternalServerError.
//
// The response body returned by this function is a Status message marshaled by a Marshaler.
//
// Do not set this function to HTTPError variable directly, use WithProtoErrorHandler option instead.
func DefaultHTTPProtoErrorHandler(ctx context.Context, mux *ServeMux, marshaler Marshaler, w http.ResponseWriter, _ *http.Request, err error) {
	// return Internal when Marshal failed
	const fallback = `{"code": 13, "message": "failed to marshal error message"}`

	s, ok := status.FromError(err)
	if !ok {
		s = status.New(codes.Unknown, err.Error())
	}

	w.Header().Del("Trailer")

	contentType := marshaler.ContentType()
	// Check marshaler on run time in order to keep backwards compatability
	// An interface param needs to be added to the ContentType() function on
	// the Marshal interface to be able to remove this check
	if httpBodyMarshaler, ok := marshaler.(*HTTPBodyMarshaler); ok {
		pb := s.Proto()
		contentType = httpBodyMarshaler.ContentTypeFromMessage(pb)
	}
	w.Header().Set("Content-Type", contentType)

	buf, merr := marshaler.Marshal(s.Proto())
	if merr != nil {
		grpclog.Infof("Failed to marshal error message %q: %v", s.Proto(), merr)
		w.WriteHeader(http.StatusInternalServerError)
		if _, err := io.WriteString(w, fallback); err != nil {
			grpclog.Infof("Failed to write response: %v", err)
		}
		return
	}

	md, ok := ServerMetadataFromContext(ctx)
	if !ok {
		grpclog.Infof("Failed to extract ServerMetadata from context")
	}

	handleForwardResponseServerMetadata(w, mux, md)
	handleForwardResponseTrailerHeader(w, md)
	st := HTTPStatusFromCode(s.Code())
	w.WriteHeader(st)
	if _, err := w.Write(buf); err != nil {
		grpclog.Infof("Failed to write response: %v", err)
	}

	handleForwardResponseTrailer(w, md)
}

// DefaultHTTPStreamErrorHandler converts the given err into a *StreamError via
// default logic.
//
// It extracts the gRPC status from err if possible. The fields of the status are
// used to populate the returned StreamError, and the HTTP status code is derived
// from the gRPC code via HTTPStatusFromCode. If the given err does not contain a
// gRPC status, an "Unknown" gRPC code is used and "Internal Server Error" HTTP code.
func DefaultHTTPStreamErrorHandler(_ context.Context, err error) *StreamError {
	grpcCode := codes.Unknown
	grpcMessage := err.Error()
	var grpcDetails []*any.Any
	if s, ok := status.FromError(err); ok {
		grpcCode = s.Code()
		grpcMessage = s.Message()
		grpcDetails = s.Proto().GetDetails()
	}
	httpCode := HTTPStatusFromCode(grpcCode)
	return &StreamError{
		GrpcCode:   int32(grpcCode),
		HttpCode:   int32(httpCode),
		Message:    grpcMessage,
		HttpStatus: http.StatusText(httpCode),
		Details:    grpcDetails,
	}
}

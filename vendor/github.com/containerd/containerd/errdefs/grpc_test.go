package errdefs

import (
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"

	"github.com/pkg/errors"
)

func TestGRPCRoundTrip(t *testing.T) {
	errShouldLeaveAlone := errors.New("unknown to package")

	for _, testcase := range []struct {
		input error
		cause error
		str   string
	}{
		{
			input: ErrAlreadyExists,
			cause: ErrAlreadyExists,
		},
		{
			input: ErrNotFound,
			cause: ErrNotFound,
		},
		{
			input: errors.Wrapf(ErrFailedPrecondition, "test test test"),
			cause: ErrFailedPrecondition,
			str:   "test test test: failed precondition",
		},
		{
			input: grpc.Errorf(codes.Unavailable, "should be not available"),
			cause: ErrUnavailable,
			str:   "should be not available: unavailable",
		},
		{
			input: errShouldLeaveAlone,
			cause: ErrUnknown,
			str:   errShouldLeaveAlone.Error() + ": " + ErrUnknown.Error(),
		},
	} {
		t.Run(testcase.input.Error(), func(t *testing.T) {
			t.Logf("input: %v", testcase.input)
			gerr := ToGRPC(testcase.input)
			t.Logf("grpc: %v", gerr)
			ferr := FromGRPC(gerr)
			t.Logf("recovered: %v", ferr)

			if errors.Cause(ferr) != testcase.cause {
				t.Fatalf("unexpected cause: %v != %v", errors.Cause(ferr), testcase.cause)
			}

			expected := testcase.str
			if expected == "" {
				expected = testcase.cause.Error()
			}
			if ferr.Error() != expected {
				t.Fatalf("unexpected string: %q != %q", ferr.Error(), expected)
			}
		})
	}

}

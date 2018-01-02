package libnetwork

import (
	"testing"

	"github.com/docker/libnetwork/types"
)

func TestErrorInterfaces(t *testing.T) {

	badRequestErrorList := []error{ErrInvalidID(""), ErrInvalidName(""), ErrInvalidJoin{}, ErrInvalidNetworkDriver(""), InvalidContainerIDError(""), ErrNoSuchNetwork(""), ErrNoSuchEndpoint("")}
	for _, err := range badRequestErrorList {
		switch u := err.(type) {
		case types.BadRequestError:
			return
		default:
			t.Fatalf("Failed to detect err %v is of type BadRequestError. Got type: %T", err, u)
		}
	}

	maskableErrorList := []error{ErrNoContainer{}}
	for _, err := range maskableErrorList {
		switch u := err.(type) {
		case types.MaskableError:
			return
		default:
			t.Fatalf("Failed to detect err %v is of type MaskableError. Got type: %T", err, u)
		}
	}

	notFoundErrorList := []error{NetworkTypeError(""), &UnknownNetworkError{}, &UnknownEndpointError{}}
	for _, err := range notFoundErrorList {
		switch u := err.(type) {
		case types.NotFoundError:
			return
		default:
			t.Fatalf("Failed to detect err %v is of type NotFoundError. Got type: %T", err, u)
		}
	}

	forbiddenErrorList := []error{NetworkTypeError(""), &UnknownNetworkError{}, &UnknownEndpointError{}}
	for _, err := range forbiddenErrorList {
		switch u := err.(type) {
		case types.ForbiddenError:
			return
		default:
			t.Fatalf("Failed to detect err %v is of type ForbiddenError. Got type: %T", err, u)
		}
	}

}

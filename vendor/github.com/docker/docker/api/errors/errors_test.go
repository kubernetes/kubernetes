package errors

import (
	"fmt"
	"github.com/stretchr/testify/assert"
	"net/http"
	"testing"
)

func newError(errorname string) error {

	return fmt.Errorf("test%v", errorname)
}

func TestErrors(t *testing.T) {
	errmsg := newError("apiError")
	err := apiError{
		error:      errmsg,
		statusCode: 0,
	}
	assert.Equal(t, err.HTTPErrorStatusCode(), err.statusCode)

	errmsg = newError("ErrorWithStatusCode")
	errcode := 1
	serr := NewErrorWithStatusCode(errmsg, errcode)
	apierr, ok := serr.(apiError)
	if !ok {
		t.Fatal("excepted err is apiError type")
	}
	assert.Equal(t, errcode, apierr.statusCode)

	errmsg = newError("NewBadRequestError")
	baderr := NewBadRequestError(errmsg)
	apierr, ok = baderr.(apiError)
	if !ok {
		t.Fatal("excepted err is apiError type")
	}
	assert.Equal(t, http.StatusBadRequest, apierr.statusCode)

	errmsg = newError("RequestForbiddenError")
	ferr := NewRequestForbiddenError(errmsg)
	apierr, ok = ferr.(apiError)
	if !ok {
		t.Fatal("excepted err is apiError type")
	}
	assert.Equal(t, http.StatusForbidden, apierr.statusCode)

	errmsg = newError("RequestNotFoundError")
	nerr := NewRequestNotFoundError(errmsg)
	apierr, ok = nerr.(apiError)
	if !ok {
		t.Fatal("excepted err is apiError type")
	}
	assert.Equal(t, http.StatusNotFound, apierr.statusCode)

	errmsg = newError("RequestConflictError")
	cerr := NewRequestConflictError(errmsg)
	apierr, ok = cerr.(apiError)
	if !ok {
		t.Fatal("excepted err is apiError type")
	}
	assert.Equal(t, http.StatusConflict, apierr.statusCode)

}

package promise

import (
	"errors"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestGo(t *testing.T) {
	errCh := Go(functionWithError)
	er := <-errCh
	require.EqualValues(t, "Error Occurred", er.Error())

	noErrCh := Go(functionWithNoError)
	er = <-noErrCh
	require.Nil(t, er)
}

func functionWithError() (err error) {
	return errors.New("Error Occurred")
}
func functionWithNoError() (err error) {
	return nil
}

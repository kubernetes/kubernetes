// +build windows

package remote

import (
	"errors"
	"os"
)

func NewOutputInterceptor() OutputInterceptor {
	return &outputInterceptor{}
}

type outputInterceptor struct {
	intercepting bool
}

func (interceptor *outputInterceptor) StartInterceptingOutput() error {
	if interceptor.intercepting {
		return errors.New("Already intercepting output!")
	}
	interceptor.intercepting = true

	// not working on windows...

	return nil
}

func (interceptor *outputInterceptor) StopInterceptingAndReturnOutput() (string, error) {
	// not working on windows...
	interceptor.intercepting = false

	return "", nil
}

func (interceptor *outputInterceptor) StreamTo(*os.File) {}

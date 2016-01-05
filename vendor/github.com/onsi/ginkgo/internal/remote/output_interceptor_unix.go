// +build freebsd openbsd netbsd dragonfly darwin linux

package remote

import (
	"errors"
	"io/ioutil"
	"os"
	"syscall"
)

func NewOutputInterceptor() OutputInterceptor {
	return &outputInterceptor{}
}

type outputInterceptor struct {
	redirectFile *os.File
	intercepting bool
}

func (interceptor *outputInterceptor) StartInterceptingOutput() error {
	if interceptor.intercepting {
		return errors.New("Already intercepting output!")
	}
	interceptor.intercepting = true

	var err error

	interceptor.redirectFile, err = ioutil.TempFile("", "ginkgo-output")
	if err != nil {
		return err
	}

	syscall.Dup2(int(interceptor.redirectFile.Fd()), 1)
	syscall.Dup2(int(interceptor.redirectFile.Fd()), 2)

	return nil
}

func (interceptor *outputInterceptor) StopInterceptingAndReturnOutput() (string, error) {
	if !interceptor.intercepting {
		return "", errors.New("Not intercepting output!")
	}

	interceptor.redirectFile.Close()
	output, err := ioutil.ReadFile(interceptor.redirectFile.Name())
	os.Remove(interceptor.redirectFile.Name())

	interceptor.intercepting = false

	return string(output), err
}

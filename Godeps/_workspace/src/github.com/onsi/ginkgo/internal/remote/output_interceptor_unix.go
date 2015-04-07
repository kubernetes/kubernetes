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
	stdoutPlaceholder *os.File
	stderrPlaceholder *os.File
	redirectFile      *os.File
	intercepting      bool
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

	interceptor.stdoutPlaceholder, err = ioutil.TempFile("", "ginkgo-output")
	if err != nil {
		return err
	}

	interceptor.stderrPlaceholder, err = ioutil.TempFile("", "ginkgo-output")
	if err != nil {
		return err
	}

	syscall.Dup2(1, int(interceptor.stdoutPlaceholder.Fd()))
	syscall.Dup2(2, int(interceptor.stderrPlaceholder.Fd()))

	syscall.Dup2(int(interceptor.redirectFile.Fd()), 1)
	syscall.Dup2(int(interceptor.redirectFile.Fd()), 2)

	return nil
}

func (interceptor *outputInterceptor) StopInterceptingAndReturnOutput() (string, error) {
	if !interceptor.intercepting {
		return "", errors.New("Not intercepting output!")
	}

	syscall.Dup2(int(interceptor.stdoutPlaceholder.Fd()), 1)
	syscall.Dup2(int(interceptor.stderrPlaceholder.Fd()), 2)

	for _, f := range []*os.File{interceptor.redirectFile, interceptor.stdoutPlaceholder, interceptor.stderrPlaceholder} {
		f.Close()
	}

	output, err := ioutil.ReadFile(interceptor.redirectFile.Name())

	for _, f := range []*os.File{interceptor.redirectFile, interceptor.stdoutPlaceholder, interceptor.stderrPlaceholder} {
		os.Remove(f.Name())
	}

	interceptor.intercepting = false

	return string(output), err
}

// +build linux freebsd

package devices

import (
	"errors"
	"os"
	"testing"

	"golang.org/x/sys/unix"
)

func TestDeviceFromPathLstatFailure(t *testing.T) {
	testError := errors.New("test error")

	// Override unix.Lstat to inject error.
	unixLstat = func(path string, stat *unix.Stat_t) error {
		return testError
	}

	_, err := DeviceFromPath("", "")
	if err != testError {
		t.Fatalf("Unexpected error %v, expected %v", err, testError)
	}
}

func TestHostDevicesIoutilReadDirFailure(t *testing.T) {
	testError := errors.New("test error")

	// Override ioutil.ReadDir to inject error.
	ioutilReadDir = func(dirname string) ([]os.FileInfo, error) {
		return nil, testError
	}

	_, err := HostDevices()
	if err != testError {
		t.Fatalf("Unexpected error %v, expected %v", err, testError)
	}
}

func TestHostDevicesIoutilReadDirDeepFailure(t *testing.T) {
	testError := errors.New("test error")
	called := false

	// Override ioutil.ReadDir to inject error after the first call.
	ioutilReadDir = func(dirname string) ([]os.FileInfo, error) {
		if called {
			return nil, testError
		}
		called = true

		// Provoke a second call.
		fi, err := os.Lstat("/tmp")
		if err != nil {
			t.Fatalf("Unexpected error %v", err)
		}

		return []os.FileInfo{fi}, nil
	}

	_, err := HostDevices()
	if err != testError {
		t.Fatalf("Unexpected error %v, expected %v", err, testError)
	}
}

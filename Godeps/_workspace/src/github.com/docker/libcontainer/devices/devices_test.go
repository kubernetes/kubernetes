package devices

import (
	"errors"
	"os"
	"testing"
)

func TestGetDeviceLstatFailure(t *testing.T) {
	testError := errors.New("test error")

	// Override os.Lstat to inject error.
	osLstat = func(path string) (os.FileInfo, error) {
		return nil, testError
	}

	_, err := GetDevice("", "")
	if err != testError {
		t.Fatalf("Unexpected error %v, expected %v", err, testError)
	}
}

func TestGetHostDeviceNodesIoutilReadDirFailure(t *testing.T) {
	testError := errors.New("test error")

	// Override ioutil.ReadDir to inject error.
	ioutilReadDir = func(dirname string) ([]os.FileInfo, error) {
		return nil, testError
	}

	_, err := GetHostDeviceNodes()
	if err != testError {
		t.Fatalf("Unexpected error %v, expected %v", err, testError)
	}
}

func TestGetHostDeviceNodesIoutilReadDirDeepFailure(t *testing.T) {
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

	_, err := GetHostDeviceNodes()
	if err != testError {
		t.Fatalf("Unexpected error %v, expected %v", err, testError)
	}
}

package etcd3retry

import (
	"context"
	"fmt"
	"net"
	"net/url"
	"strconv"
	"syscall"
	"testing"

	etcdrpc "go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	"k8s.io/apiserver/pkg/storage"
)

func TestOnError(t *testing.T) {
	tests := []struct {
		name               string
		retriableFn        func(error) (string, bool)
		returnedFnError    func(retryCounter int) error
		expectedRetries    int
		expectedFinalError error
	}{
		{
			name:               "retry ErrLeaderChanged",
			retriableFn:        IsRetriableErrorOnRead,
			returnedFnError:    func(_ int) error { return etcdrpc.ErrLeaderChanged },
			expectedRetries:    5,
			expectedFinalError: etcdrpc.ErrLeaderChanged,
		},
		{
			name:        "retry ErrLeaderChanged a few times",
			retriableFn: IsRetriableErrorOnRead,
			returnedFnError: func(retryCounter int) error {
				if retryCounter == 3 {
					return nil
				}
				return etcdrpc.ErrLeaderChanged
			},
			expectedRetries: 3,
		},
		{
			name:            "no retries",
			retriableFn:     IsRetriableErrorOnRead,
			returnedFnError: func(_ int) error { return nil },
		},
		{
			name:               "no retries for a random error",
			retriableFn:        IsRetriableErrorOnRead,
			returnedFnError:    func(_ int) error { return fmt.Errorf("random error") },
			expectedFinalError: fmt.Errorf("random error"),
		},
	}

	for _, scenario := range tests {
		t.Run(scenario.name, func(t *testing.T) {
			ctx := context.TODO()
			// we set it to -1 to indicate that the first
			// execution is not a retry
			actualRetries := -1
			err := OnError(ctx, DefaultRetry, scenario.retriableFn, func() error {
				actualRetries++
				return scenario.returnedFnError(actualRetries)
			})

			if actualRetries != scenario.expectedRetries {
				t.Errorf("Unexpected number of retries %v, expected %v", actualRetries, scenario.expectedRetries)
			}
			if (err == nil && scenario.expectedFinalError != nil) || (err != nil && scenario.expectedFinalError == nil) {
				t.Errorf("Expected error %v, got %v", scenario.expectedFinalError, err)
			}
			if err != nil && scenario.expectedFinalError != nil && err.Error() != scenario.expectedFinalError.Error() {
				t.Errorf("Expected error %v, got %v", scenario.expectedFinalError, err)
			}
		})
	}
}

func TestIsRetriableErrorOnRead(t *testing.T) {
	tests := []struct {
		name               string
		etcdErr            error
		errorLabelExpected string
		retryExpected      bool
	}{
		{
			name:               "generic storage error",
			etcdErr:            storage.NewKeyNotFoundError("key", 0),
			errorLabelExpected: "",
			retryExpected:      false,
		},
		{
			name:               "connection refused error",
			etcdErr:            &url.Error{Err: &net.OpError{Err: syscall.ECONNREFUSED}},
			errorLabelExpected: "",
			retryExpected:      false,
		},
		{
			name:               "etcd unavailable error",
			etcdErr:            etcdrpc.ErrLeaderChanged,
			errorLabelExpected: "LeaderChanged",
			retryExpected:      true,
		},
		{
			name:               "should also inspect error message",
			etcdErr:            fmt.Errorf("etcdserver: no leader"),
			errorLabelExpected: "NoLeader",
			retryExpected:      true,
		},
		{
			name:               "unavailable code with unrecognized suffix",
			etcdErr:            etcdrpc.ErrGRPCUnhealthy,
			errorLabelExpected: "Unavailable",
			retryExpected:      true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errorCodeGot, retryGot := IsRetriableErrorOnRead(test.etcdErr)

			if test.errorLabelExpected != errorCodeGot {
				t.Errorf("expected error code: %s  but got: %s", test.errorLabelExpected, errorCodeGot)
			}

			if test.retryExpected != retryGot {
				t.Errorf("expected retry: %s  but got: %s", strconv.FormatBool(test.retryExpected), strconv.FormatBool(retryGot))
			}
		})
	}
}

func TestIsRetriableErrorOnWrite(t *testing.T) {
	tests := []struct {
		name               string
		etcdErr            error
		errorLabelExpected string
		retryExpected      bool
	}{
		{
			name:               "generic storage error",
			etcdErr:            storage.NewKeyNotFoundError("key", 0),
			errorLabelExpected: "",
			retryExpected:      false,
		},
		{
			name:               "connection refused error",
			etcdErr:            &url.Error{Err: &net.OpError{Err: syscall.ECONNREFUSED}},
			errorLabelExpected: "",
			retryExpected:      false,
		},
		{
			name:               "etcd unavailable error",
			etcdErr:            etcdrpc.ErrLeaderChanged,
			errorLabelExpected: "LeaderChanged",
			retryExpected:      true,
		},
		{
			name:               "should also inspect error message",
			etcdErr:            fmt.Errorf("etcdserver: no leader"),
			errorLabelExpected: "NoLeader",
			retryExpected:      true,
		},
		{
			name:               "unavailable code with unrecognized suffix",
			etcdErr:            etcdrpc.ErrGRPCUnhealthy,
			errorLabelExpected: "",
			retryExpected:      false,
		},
		{
			name:               "timeout not retried for writes",
			etcdErr:            etcdrpc.ErrGRPCTimeout,
			errorLabelExpected: "",
			retryExpected:      false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errorCodeGot, retryGot := IsRetriableErrorOnWrite(test.etcdErr)

			if test.errorLabelExpected != errorCodeGot {
				t.Errorf("expected error code: %s  but got: %s", test.errorLabelExpected, errorCodeGot)
			}

			if test.retryExpected != retryGot {
				t.Errorf("expected retry: %s  but got: %s", strconv.FormatBool(test.retryExpected), strconv.FormatBool(retryGot))
			}
		})
	}
}

package etcd3retry

import (
	"fmt"
	"net"
	"net/url"
	"strconv"
	"syscall"
	"testing"

	etcdrpc "go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	"k8s.io/apiserver/pkg/storage"
)

func TestIsRetriableEtcdError(t *testing.T) {
	tests := []struct {
		name               string
		etcdErr            error
		errorLabelExpected string
		retryExpected      bool
	}{
		{
			name:               "error is nil",
			errorLabelExpected: "",
			retryExpected:      false,
		},
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
			errorLabelExpected: "Unavailable",
			retryExpected:      true,
		},
		{
			name:               "should also inspect error message",
			etcdErr:            fmt.Errorf("etcdserver: leader changed"),
			errorLabelExpected: "Unavailable",
			retryExpected:      true,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errorCodeGot, retryGot := isRetriableEtcdError(test.etcdErr)

			if test.errorLabelExpected != errorCodeGot {
				t.Errorf("expected error code: %s  but got: %s", test.errorLabelExpected, errorCodeGot)
			}

			if test.retryExpected != retryGot {
				t.Errorf("expected retry: %s  but got: %s", strconv.FormatBool(test.retryExpected), strconv.FormatBool(retryGot))
			}
		})
	}
}

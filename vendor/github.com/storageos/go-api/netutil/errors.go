package netutil

import (
	"errors"
	"fmt"
	"strings"

	"github.com/storageos/go-api/serror"
)

// ErrAllFailed produces a typed StorageOS error which should be used to indicate that
// the API is not contactable for all of the supplied node addresses.
func ErrAllFailed(addrs []string) error {
	msg := fmt.Sprintf("failed to dial all known cluster members, (%s)", strings.Join(addrs, ","))
	help := "ensure that the value of $STORAGEOS_HOST (or the -H flag) is correct, and that there are healthy StorageOS nodes in this cluster"

	return serror.NewTypedStorageOSError(serror.APIUncontactable, nil, msg, help)
}

func newInvalidNodeError(err error) error {
	msg := fmt.Sprintf("invalid node format: %s", err)
	help := "please check the format of $STORAGEOS_HOST (or the -H flag) complies with the StorageOS JOIN format"

	return serror.NewTypedStorageOSError(serror.InvalidHostConfig, err, msg, help)
}

var (
	errUnsupportedScheme = errors.New("unsupported URL scheme")
	errInvalidHostName   = errors.New("invalid hostname")
	errInvalidPortNumber = errors.New("invalid port number")
)

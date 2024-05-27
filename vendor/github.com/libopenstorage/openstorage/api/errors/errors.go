package errors

import (
	"fmt"
	"strings"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/pkg/parser"
)

var (
	// ErrResourceNotInitialized is error string to return when SDK resources are not yet initialized
	ErrResourceNotInitialized = fmt.Errorf("resource has not been initialized")
)

// ErrNotFound error type for objects not found
type ErrNotFound struct {
	// ID unique object identifier.
	ID string
	// Type of the object which wasn't found
	Type string
}

func (e *ErrNotFound) Error() string {
	return fmt.Sprintf("%v with ID: %v not found", e.Type, e.ID)
}

// ErrExists type for objects already present
type ErrExists struct {
	// ID unique object identifier.
	ID string
	// Type of the object which already exists
	Type string
}

func (e *ErrExists) Error() string {
	return fmt.Sprintf("%v with ID: %v already exists", e.Type, e.ID)
}

// ErrNotSupported error type for APIs that are not supported
type ErrNotSupported struct{}

func (e *ErrNotSupported) Error() string {
	return fmt.Sprintf("Not Supported")
}

// ErrStoragePoolExpandInProgress error when an expand is already in progress
// on a storage pool
type ErrStoragePoolResizeInProgress struct {
	// Pool is the affected pool
	Pool *api.StoragePool
}

func (e *ErrStoragePoolResizeInProgress) Error() string {
	errMsg := fmt.Sprintf("resize for pool %s is already in progress.", e.Pool.GetUuid())
	if e.Pool.LastOperation != nil {
		op := e.Pool.LastOperation
		if op.Type == api.SdkStoragePool_OPERATION_RESIZE {
			errMsg = fmt.Sprintf("%s %s %s", errMsg, op.Msg, parser.LabelsToString(op.Params))
		}
	}

	return errMsg
}

func IsAccessDenied(err error) bool {
	if err == nil {
		return false
	}

	if strings.Contains(err.Error(), "Access denied") {
		return true
	}

	return false
}

// ErrCloudBackupServerBusy indicates server being busy with too many
// requests
type ErrCloudBackupServerBusy struct {
	// object ID for which server is busy
	ID string
}

func (e *ErrCloudBackupServerBusy) Error() string {
	if len(e.ID) > 0 {
		return fmt.Sprintf("cloud backup server busy: concurrent backup limit for volume %v reached", e.ID)
	}
	return fmt.Sprintf("cloud backup server busy")
}

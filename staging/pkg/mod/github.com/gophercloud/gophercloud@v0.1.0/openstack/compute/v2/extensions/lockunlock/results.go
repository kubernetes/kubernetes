package lockunlock

import (
	"github.com/gophercloud/gophercloud"
)

// LockResult and UnlockResult are the responses from a Lock and Unlock
// operations respectively. Call their ExtractErr methods to determine if the
// requests suceeded or failed.
type LockResult struct {
	gophercloud.ErrResult
}

type UnlockResult struct {
	gophercloud.ErrResult
}

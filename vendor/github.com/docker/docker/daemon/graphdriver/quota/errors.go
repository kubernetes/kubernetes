package quota

import "github.com/docker/docker/api/errdefs"

var (
	_ errdefs.ErrNotImplemented = (*errQuotaNotSupported)(nil)
)

// ErrQuotaNotSupported indicates if were found the FS didn't have projects quotas available
var ErrQuotaNotSupported = errQuotaNotSupported{}

type errQuotaNotSupported struct {
}

func (e errQuotaNotSupported) NotImplemented() {}

func (e errQuotaNotSupported) Error() string {
	return "Filesystem does not support, or has not enabled quotas"
}

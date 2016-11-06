package native

import (
	"k8s.io/kubernetes/pkg/types"
)

type itemData struct {
	uid    types.UID
	data   []byte
	expiry uint64
	lsn    LSN
}

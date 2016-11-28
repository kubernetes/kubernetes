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

// ByName allows sorting of ItemData by the Path field
type ByPath []*ItemData

func (a ByPath) Len() int           { return len(a) }
func (a ByPath) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByPath) Less(i, j int) bool { return a[i].Path < a[j].Path }

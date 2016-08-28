package utils

import (
	"sort"

	"github.com/emccode/libstorage/api/types"
)

// ByVolumeID implements sort.Interface for []*types.Volume based on the ID
// field.
type ByVolumeID []*types.Volume

func (a ByVolumeID) Len() int           { return len(a) }
func (a ByVolumeID) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByVolumeID) Less(i, j int) bool { return a[i].ID < a[j].ID }

// SortVolumeByID sorts the volumes by their IDs.
func SortVolumeByID(volumes []*types.Volume) []*types.Volume {
	sort.Sort(ByVolumeID(volumes))
	return volumes
}

// ByString  implements sort.Interface for []string.
type ByString []string

func (a ByString) Len() int           { return len(a) }
func (a ByString) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a ByString) Less(i, j int) bool { return a[i] < a[j] }

// SortByString sorts the strings.
func SortByString(strings []string) []string {
	sort.Sort(ByString(strings))
	return strings
}

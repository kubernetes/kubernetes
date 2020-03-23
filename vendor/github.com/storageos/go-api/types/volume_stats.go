package types

// VolumeStats - volume stats (volume counts, looking forward to capacity)
type VolumeStats struct {
	MasterVolumeCount  int `json:"masterVolumeCount"`
	ReplicaVolumeCount int `json:"replicaVolumeCount"`
	VirtualVolumeCount int `json:"virtualVolumeCount"`
}

package quobyte

type createVolumeRequest struct {
	Name        string `json:"name"`
	RootUserID  string `json:"root_user_id"`
	RootGroupID string `json:"root_group_id"`
}

type createVolumeResponse struct {
	VolumeUUID string `json:"volume_uuid"`
}

type deleteVolumeRequest struct {
	VolumeUUID string `json:"volume_uuid"`
}

type deleteVolumeResponse struct {
}

type resolveVolumeNameRequest struct {
	VolumeName string `json:"volume_name,omitempty"`
}

type resolveVolumeNameResponse struct {
	VolumeUUID string `json:"volume_uuid,omitempty"`
}

package volume

import (
	"time"
)

type Volume struct {
	Status           string      `json:"status"`
	Description      interface{} `json:"description"`
	VolumeName       string      `json:"volume_name"`
	SubCode          int         `json:"sub_code"`
	TransitionStatus string      `json:"transition_status"`
	Instance         struct {
		InstanceID   string `json:"instance_id"`
		InstanceName string `json:"instance_name"`
		Device       string `json:"device"`
	} `json:"instance"`
	CreateTime time.Time `json:"create_time"`
	VolumeID   string    `json:"volume_id"`
	StatusTime time.Time `json:"status_time"`
	Size       int       `json:"size"`
}

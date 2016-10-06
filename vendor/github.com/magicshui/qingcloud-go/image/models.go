package image

import (
	"time"
)

type Image struct {
	Status           string      `json:"status"`
	ProcessorType    string      `json:"processor_type"`
	ImageID          string      `json:"image_id"`
	SubCode          int         `json:"sub_code"`
	TransitionStatus string      `json:"transition_status"`
	RecommendedType  string      `json:"recommended_type"`
	ImageName        string      `json:"image_name"`
	Visibility       string      `json:"visibility"`
	Platform         string      `json:"platform"`
	CreateTime       time.Time   `json:"create_time"`
	OsFamily         string      `json:"os_family"`
	Provider         string      `json:"provider"`
	Owner            string      `json:"owner"`
	StatusTime       time.Time   `json:"status_time"`
	Size             int         `json:"size"`
	Description      interface{} `json:"description"`
}

type ImageUser struct {
	ImageID    string    `json:"image_id"`
	CreateTime time.Time `json:"create_time"`
	User       struct {
		UserID string `json:"user_id"`
		Email  string `json:"email"`
	} `json:"user"`
}

package tag

import (
	"time"
)

type Tag struct {
	TagID             string `json:"tag_id"`
	TagName           string `json:"tag_name"`
	Description       string `json:"description"`
	ResourceCount     int    `json:"resource_count"`
	ResourceTypeCount []struct {
		Count        int    `json:"count"`
		ResourceType string `json:"resource_type"`
	} `json:"resource_type_count"`
	ResourceTagPairs []struct {
		TagID        string `json:"tag_id"`
		ResourceType string `json:"resource_type"`
		ResourceID   string `json:"resource_id"`
	} `json:"resource_tag_pairs"`
	CreateTime time.Time `json:"create_time"`
}

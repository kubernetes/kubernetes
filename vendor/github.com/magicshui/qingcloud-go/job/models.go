package job

import (
	"time"
)

type Job struct {
	JobID       string    `json:"job_id"`
	JobAction   string    `json:"job_action"`
	CreateTime  time.Time `json:"create_time"`
	ResourceIds string    `json:"resource_ids"`
	Owner       string    `json:"owner"`
	ErrorCodes  string    `json:"error_codes"`
	Status      string    `json:"status"`
	StatusTime  time.Time `json:"status_time"`
}

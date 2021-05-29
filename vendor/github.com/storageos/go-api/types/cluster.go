package types

import "time"

// Maintenance is used to place the cluster in maintenance mode.
type Maintenance struct {
	Enabled   bool      `json:"enabled"`
	UpdatedBy string    `json:"updatedBy"`
	UpdatedAt time.Time `json:"updatedAt"`
}

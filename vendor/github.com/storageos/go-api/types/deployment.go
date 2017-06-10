package types

import "time"

// Deployment Volume master or replica deployment details.
// swagger:model Deployment
type Deployment struct {

	// Deployment unique ID
	// Read Only: true
	ID string `json:"id"`

	// Inode number
	// Read Only: true
	Inode uint32 `json:"inode"`

	// Controller ID
	// Read Only: true
	Controller string `json:"controller"`

	// Health
	// Read Only: true
	Health string `json:"health"`

	// Status
	// Read Only: true
	Status string `json:"status"`

	// Created at
	// Read Only: true
	CreatedAt time.Time `json:"createdAt"`
}

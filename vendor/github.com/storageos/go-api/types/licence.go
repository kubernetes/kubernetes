package types

import "time"

// FeatureType store features types
type FeatureType string

const (
	// HA means High Availability
	HA = FeatureType("HA")
	// DEV means developer licence
	DEV = FeatureType("DEV")
	// TRIAL means trial licence
	TRIAL = FeatureType("TRIAL")
)

// Licence holds the information to be encoded in the licence key. It needs to be synced across
// the django server running on portal-API as well as the corresponding decoding package on the
// storageOS control plane
type Licence struct {
	ArrayUUID    string               `json:"arrayUUID,omitempty"`
	ClusterID    string               `json:"clusterID,omitempty"`
	CustomerID   string               `json:"customerID"`
	CustomerName string               `json:"customerName"`
	Storage      int                  `json:"storage"`
	ValidUntil   time.Time            `json:"validUntil"`
	LicenceType  string               `json:"licenceType"`
	Features     map[FeatureType]bool `json:"features"`
	Unregistered bool                 `json:"unregistered"`
}

// LicenceKeyContainer - stores a licence key
type LicenceKeyContainer struct {
	Key string `json:"key"`
}

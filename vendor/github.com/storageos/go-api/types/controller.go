package types

// Versions and Prefixes used in API and KV URLs
import "time"

const (
	ControllerAPIPrefix     = "controller"
	ControllerDefaultPort   = "3260"
	ControllerScanAPIPrefix = "config/scan"
)

// ControllerCurrent - current controller
var ControllerCurrent = ""

// Controller status phases
const (
	ControllerStatusPending    = "pending"
	ControllerStatusEvaluating = "evaluating"
	ControllerStatusDeploying  = "deploying"
	ControllerStatusActive     = "active"
	ControllerStatusFailed     = "failed"
	ControllerStatusDeleting   = "deleting"

	ControllerHealthStarting = "starting"
	ControllerHealthOK       = "healthy"
	ControllerHealthDegraded = "degraded"
	ControllerHealthOffline  = "offline"
)

// Errors for controller related things
const (
	ErrControllerHostIDAllocation string = "error, could not allocate hostid"
	ErrControllerIDNotSet                = "error, controller ID not set"
	ErrControllerNotFound                = "controller not found"
)

// Controller is used to represent a storage node in a cluster
type Controller struct {
	ID               string                 `json:"id,omitempty"`
	HostID           uint16                 `json:"hostID"`
	Scheduler        bool                   `json:"scheduler"`
	Name             string                 `json:"name"`
	Address          string                 `json:"address"`
	APIPort          int                    `json:"apiPort"`
	NatsPort         int                    `json:"natsPort"`
	NatsClusterPort  int                    `json:"natsClusterPort"`
	SerfPort         int                    `json:"serfPort"`
	DFSPort          int                    `json:"dfsPort"`
	Description      string                 `json:"description"`
	ControllerGroups []string               `json:"controllerGroups"`
	Tags             []string               `json:"tags"`
	Labels           map[string]string      `json:"labels"`
	VolumeStats      VolumeStats            `json:"volumeStats"`
	PoolStats        map[string]DriverStats `json:"poolStats"`

	// health is updated by the
	Health          string                 `json:"health"`
	HealthUpdatedAt time.Time              `json:"healthUpdatedAt"`
	VersionInfo     map[string]VersionInfo `json:"versionInfo"`
	Version         string                 `json:"version"`

	// high level stats that combine info from all driver instances
	CapacityStats CapacityStats `json:"capacityStats"`
}

// DriverStats is used to report stats for all drivers in a pool.
type DriverStats map[string]CapacityStats

// VolumeStats - volume stats (volume counts, looking forward to capacity)
type VolumeStats struct {
	MasterVolumeCount  int `json:"masterVolumeCount"`
	ReplicaVolumeCount int `json:"replicaVolumeCount"`
	VirtualVolumeCount int `json:"virtualVolumeCount"`
}

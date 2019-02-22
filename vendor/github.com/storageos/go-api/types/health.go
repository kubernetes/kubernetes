package types

import "encoding/json"

type SubModuleStatus struct {
	Status    string `json:"status"`
	UpdatedAt string `json:"updatedAt"`
	ChangedAt string `json:"changedAt"`
	Message   string `json:"message"`
}

type NamedSubModuleStatus struct {
	Name string
	SubModuleStatus
}

type CPHealthStatus struct {
	KV        SubModuleStatus
	KVWrite   SubModuleStatus
	NATS      SubModuleStatus
	Scheduler SubModuleStatus
}

func (c *CPHealthStatus) ToNamedSubmodules() []NamedSubModuleStatus {
	return []NamedSubModuleStatus{
		{Name: "nats", SubModuleStatus: c.NATS},
		{Name: "kv", SubModuleStatus: c.KV},
		{Name: "kv_write", SubModuleStatus: c.KVWrite},
		{Name: "scheduler", SubModuleStatus: c.Scheduler},
	}
}

func (c *CPHealthStatus) UnmarshalJSON(data []byte) error {
	unmarsh := struct {
		Submodules struct {
			KV        SubModuleStatus `json:"kv"`
			KVWrite   SubModuleStatus `json:"kv_write"`
			NATS      SubModuleStatus `json:"nats"`
			Scheduler SubModuleStatus `json:"scheduler"`
		} `json:"submodules"`
	}{}

	if err := json.Unmarshal(data, &unmarsh); err != nil {
		return err
	}

	c.KV = unmarsh.Submodules.KV
	c.KVWrite = unmarsh.Submodules.KVWrite
	c.NATS = unmarsh.Submodules.NATS
	c.Scheduler = unmarsh.Submodules.Scheduler

	return nil
}

type DPHealthStatus struct {
	DirectFSClient SubModuleStatus
	DirectFSServer SubModuleStatus
	Director       SubModuleStatus
	FSDriver       SubModuleStatus
	FS             SubModuleStatus
}

func (d *DPHealthStatus) ToNamedSubmodules() []NamedSubModuleStatus {
	return []NamedSubModuleStatus{
		{Name: "dfs_client", SubModuleStatus: d.DirectFSClient},
		{Name: "dfs_server", SubModuleStatus: d.DirectFSServer},
		{Name: "director", SubModuleStatus: d.Director},
		{Name: "fs_driver", SubModuleStatus: d.FSDriver},
		{Name: "fs", SubModuleStatus: d.FS},
	}
}

func (d *DPHealthStatus) UnmarshalJSON(data []byte) error {
	unmarsh := struct {
		Submodules struct {
			DirectFSClient SubModuleStatus `json:"directfs-client"`
			DirectFSServer SubModuleStatus `json:"directfs-server"`
			Director       SubModuleStatus `json:"director"`
			FSDriver       SubModuleStatus `json:"filesystem-driver"`
			FS             SubModuleStatus `json:"fs"`
		} `json:"submodules"`
	}{}

	if err := json.Unmarshal(data, &unmarsh); err != nil {
		return err
	}

	d.DirectFSClient = unmarsh.Submodules.DirectFSClient
	d.DirectFSServer = unmarsh.Submodules.DirectFSServer
	d.Director = unmarsh.Submodules.Director
	d.FSDriver = unmarsh.Submodules.FSDriver
	d.FS = unmarsh.Submodules.FS

	return nil
}

// HealthStatus is the health status json object.
type HealthStatus struct {
	Submodules HealthSubmodules `json:"submodules"`
}

// HealthSubmodules is the "submodules" attribuet of HealthStatus.
type HealthSubmodules struct {
	KV             SubModuleStatus `json:"kv,omitempty"`
	KVWrite        SubModuleStatus `json:"kv_write,omitempty"`
	NATS           SubModuleStatus `json:"nats,omitempty"`
	Scheduler      SubModuleStatus `json:"scheduler,omitempty"`
	DirectFSClient SubModuleStatus `json:"directfs_initiator,omitempty"`
	DirectFSServer SubModuleStatus `json:"directfs_responder,omitempty"`
	Director       SubModuleStatus `json:"director,omitempty"`
	FSDriver       SubModuleStatus `json:"rdb,omitempty"`
	FS             SubModuleStatus `json:"presentation,omitempty"`
}

// ToCPHealthStatus returns only CPHealthStatus from the HealthStatus.
func (h *HealthStatus) ToCPHealthStatus() *CPHealthStatus {
	return &CPHealthStatus{
		KV:        h.Submodules.KV,
		KVWrite:   h.Submodules.KVWrite,
		NATS:      h.Submodules.KVWrite,
		Scheduler: h.Submodules.Scheduler,
	}
}

// ToDPHealthStatus returns only DPHealthStatus from the HealthStatus.
func (h *HealthStatus) ToDPHealthStatus() *DPHealthStatus {
	return &DPHealthStatus{
		DirectFSClient: h.Submodules.DirectFSClient,
		DirectFSServer: h.Submodules.DirectFSServer,
		Director:       h.Submodules.Director,
		FSDriver:       h.Submodules.FSDriver,
		FS:             h.Submodules.FS,
	}
}

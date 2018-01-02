package diskconfig

type ServerDiskConfigExt struct {
	// DiskConfig is the disk configuration of the server.
	DiskConfig DiskConfig `json:"OS-DCF:diskConfig"`
}

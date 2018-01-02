package mount

// CustomLoad defines the mounter.Load callback function for customer mounters
type CustomLoad func([]string, DeviceMap, PathMap) error
// CustomReload defines the mounter.Reload callback function for customer mounters
type CustomReload func(string, DeviceMap, PathMap) error
// CustomMounter defines the CustomMount function that retursn the load and reload callbacks
type CustomMounter func() (CustomLoad, CustomReload)

// CustomMounterHandler implements the Mounter interface
type CustomMounterHandler struct {
	Mounter
	cl CustomLoad
	cr CustomReload
}

// NewCustomMounter returns a new CustomMounter
func NewCustomMounter(
	devPrefixes []string,
	mountImpl MountImpl,
	customMounter CustomMounter,
	allowedDirs []string,
) (*CustomMounterHandler, error) {

	m := &CustomMounterHandler{
		Mounter: Mounter{
			mountImpl: mountImpl,
			mounts:    make(DeviceMap),
			paths:     make(PathMap),
			allowedDirs: allowedDirs,
		},
	}
	cl, cr := customMounter()
	m.cl = cl
	m.cr = cr
	err := m.Load(devPrefixes)
	if err != nil {
		return nil, err
	}
	return m, nil
}

// Load mount table
func (c *CustomMounterHandler) Load(devPrefixes []string) error {
	return c.cl(devPrefixes, c.mounts, c.paths)
}

// Reload mount table for a device
func (c *CustomMounterHandler) Reload(device string) error {
	return c.cr(device, c.mounts, c.paths)
}

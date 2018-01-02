package libcontainerd

const (
	// InitFriendlyName is the name given in the lookup map of processes
	// for the first process started in a container.
	InitFriendlyName = "init"
	configFilename   = "config.json"
)

type containerCommon struct {
	process
	processes map[string]*process
}

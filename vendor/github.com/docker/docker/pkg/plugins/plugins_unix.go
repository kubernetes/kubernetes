// +build !windows

package plugins

// BasePath returns the path to which all paths returned by the plugin are relative to.
// For v1 plugins, this always returns the host's root directory.
func (p *Plugin) BasePath() string {
	return "/"
}

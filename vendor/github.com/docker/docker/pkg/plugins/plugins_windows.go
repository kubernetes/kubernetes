package plugins

// BasePath returns the path to which all paths returned by the plugin are relative to.
// For Windows v1 plugins, this returns an empty string, since the plugin is already aware
// of the absolute path of the mount.
func (p *Plugin) BasePath() string {
	return ""
}

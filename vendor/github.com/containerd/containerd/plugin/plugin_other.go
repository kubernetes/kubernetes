// +build !go1.8 windows !amd64

package plugin

func loadPlugins(path string) error {
	// plugins not supported until 1.8
	return nil
}

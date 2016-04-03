package daemon

import (
	"fmt"
	"io"
)

func (daemon *Daemon) ContainerExport(name string, out io.Writer) error {
	container, err := daemon.Get(name)
	if err != nil {
		return err
	}

	data, err := container.Export()
	if err != nil {
		return fmt.Errorf("%s: %s", name, err)
	}
	defer data.Close()

	// Stream the entire contents of the container (basically a volatile snapshot)
	if _, err := io.Copy(out, data); err != nil {
		return fmt.Errorf("%s: %s", name, err)
	}
	return nil
}

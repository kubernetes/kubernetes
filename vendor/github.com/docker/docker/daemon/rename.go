package daemon

import (
	"fmt"
)

func (daemon *Daemon) ContainerRename(oldName, newName string) error {
	if oldName == "" || newName == "" {
		return fmt.Errorf("usage: docker rename OLD_NAME NEW_NAME")
	}

	container, err := daemon.Get(oldName)
	if err != nil {
		return err
	}

	oldName = container.Name

	container.Lock()
	defer container.Unlock()
	if newName, err = daemon.reserveName(container.ID, newName); err != nil {
		return fmt.Errorf("Error when allocating new name: %s", err)
	}

	container.Name = newName

	undo := func() {
		container.Name = oldName
		daemon.reserveName(container.ID, oldName)
		daemon.containerGraph.Delete(newName)
	}

	if err := daemon.containerGraph.Delete(oldName); err != nil {
		undo()
		return fmt.Errorf("Failed to delete container %q: %v", oldName, err)
	}

	if err := container.toDisk(); err != nil {
		undo()
		return err
	}

	container.LogEvent("rename")
	return nil
}

package daemon

import (
	"fmt"
	"strings"

	"github.com/docker/docker/api"
	"github.com/docker/docker/container"
	"github.com/docker/docker/pkg/namesgenerator"
	"github.com/docker/docker/pkg/stringid"
	"github.com/sirupsen/logrus"
)

var (
	validContainerNameChars   = api.RestrictedNameChars
	validContainerNamePattern = api.RestrictedNamePattern
)

func (daemon *Daemon) registerName(container *container.Container) error {
	if daemon.Exists(container.ID) {
		return fmt.Errorf("Container is already loaded")
	}
	if err := validateID(container.ID); err != nil {
		return err
	}
	if container.Name == "" {
		name, err := daemon.generateNewName(container.ID)
		if err != nil {
			return err
		}
		container.Name = name
	}
	return daemon.containersReplica.ReserveName(container.Name, container.ID)
}

func (daemon *Daemon) generateIDAndName(name string) (string, string, error) {
	var (
		err error
		id  = stringid.GenerateNonCryptoID()
	)

	if name == "" {
		if name, err = daemon.generateNewName(id); err != nil {
			return "", "", err
		}
		return id, name, nil
	}

	if name, err = daemon.reserveName(id, name); err != nil {
		return "", "", err
	}

	return id, name, nil
}

func (daemon *Daemon) reserveName(id, name string) (string, error) {
	if !validContainerNamePattern.MatchString(strings.TrimPrefix(name, "/")) {
		return "", fmt.Errorf("Invalid container name (%s), only %s are allowed", name, validContainerNameChars)
	}
	if name[0] != '/' {
		name = "/" + name
	}

	if err := daemon.containersReplica.ReserveName(name, id); err != nil {
		if err == container.ErrNameReserved {
			id, err := daemon.containersReplica.Snapshot().GetID(name)
			if err != nil {
				logrus.Errorf("got unexpected error while looking up reserved name: %v", err)
				return "", err
			}
			return "", fmt.Errorf("Conflict. The container name %q is already in use by container %q. You have to remove (or rename) that container to be able to reuse that name.", name, id)
		}
		return "", fmt.Errorf("error reserving name: %q, error: %v", name, err)
	}
	return name, nil
}

func (daemon *Daemon) releaseName(name string) {
	daemon.containersReplica.ReleaseName(name)
}

func (daemon *Daemon) generateNewName(id string) (string, error) {
	var name string
	for i := 0; i < 6; i++ {
		name = namesgenerator.GetRandomName(i)
		if name[0] != '/' {
			name = "/" + name
		}

		if err := daemon.containersReplica.ReserveName(name, id); err != nil {
			if err == container.ErrNameReserved {
				continue
			}
			return "", err
		}
		return name, nil
	}

	name = "/" + stringid.TruncateID(id)
	if err := daemon.containersReplica.ReserveName(name, id); err != nil {
		return "", err
	}
	return name, nil
}

func validateID(id string) error {
	if id == "" {
		return fmt.Errorf("Invalid empty id")
	}
	return nil
}

// +build windows

package builtin

import (
	"errors"

	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/ipam"
	"github.com/docker/libnetwork/ipamapi"
	"github.com/docker/libnetwork/ipamutils"

	windowsipam "github.com/docker/libnetwork/ipams/windowsipam"
)

// InitDockerDefault registers the built-in ipam service with libnetwork
func InitDockerDefault(ic ipamapi.Callback, l, g interface{}) error {
	var (
		ok                bool
		localDs, globalDs datastore.DataStore
	)

	if l != nil {
		if localDs, ok = l.(datastore.DataStore); !ok {
			return errors.New("incorrect local datastore passed to built-in ipam init")
		}
	}

	if g != nil {
		if globalDs, ok = g.(datastore.DataStore); !ok {
			return errors.New("incorrect global datastore passed to built-in ipam init")
		}
	}

	ipamutils.InitNetworks()

	a, err := ipam.NewAllocator(localDs, globalDs)
	if err != nil {
		return err
	}

	cps := &ipamapi.Capability{RequiresRequestReplay: true}

	return ic.RegisterIpamDriverWithCapabilities(ipamapi.DefaultIPAM, a, cps)
}

// Init registers the built-in ipam service with libnetwork
func Init(ic ipamapi.Callback, l, g interface{}) error {
	initFunc := windowsipam.GetInit(windowsipam.DefaultIPAM)

	err := InitDockerDefault(ic, l, g)
	if err != nil {
		return err
	}

	return initFunc(ic, l, g)
}

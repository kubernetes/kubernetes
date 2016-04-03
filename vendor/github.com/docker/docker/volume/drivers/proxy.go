// generated code - DO NOT EDIT

package volumedrivers

import "errors"

type client interface {
	Call(string, interface{}, interface{}) error
}

type volumeDriverProxy struct {
	client
}

type volumeDriverProxyCreateRequest struct {
	Name string
}

type volumeDriverProxyCreateResponse struct {
	Err string
}

func (pp *volumeDriverProxy) Create(name string) (err error) {
	var (
		req volumeDriverProxyCreateRequest
		ret volumeDriverProxyCreateResponse
	)

	req.Name = name
	if err = pp.Call("VolumeDriver.Create", req, &ret); err != nil {
		return
	}

	if ret.Err != "" {
		err = errors.New(ret.Err)
	}

	return
}

type volumeDriverProxyRemoveRequest struct {
	Name string
}

type volumeDriverProxyRemoveResponse struct {
	Err string
}

func (pp *volumeDriverProxy) Remove(name string) (err error) {
	var (
		req volumeDriverProxyRemoveRequest
		ret volumeDriverProxyRemoveResponse
	)

	req.Name = name
	if err = pp.Call("VolumeDriver.Remove", req, &ret); err != nil {
		return
	}

	if ret.Err != "" {
		err = errors.New(ret.Err)
	}

	return
}

type volumeDriverProxyPathRequest struct {
	Name string
}

type volumeDriverProxyPathResponse struct {
	Mountpoint string
	Err        string
}

func (pp *volumeDriverProxy) Path(name string) (mountpoint string, err error) {
	var (
		req volumeDriverProxyPathRequest
		ret volumeDriverProxyPathResponse
	)

	req.Name = name
	if err = pp.Call("VolumeDriver.Path", req, &ret); err != nil {
		return
	}

	mountpoint = ret.Mountpoint

	if ret.Err != "" {
		err = errors.New(ret.Err)
	}

	return
}

type volumeDriverProxyMountRequest struct {
	Name string
}

type volumeDriverProxyMountResponse struct {
	Mountpoint string
	Err        string
}

func (pp *volumeDriverProxy) Mount(name string) (mountpoint string, err error) {
	var (
		req volumeDriverProxyMountRequest
		ret volumeDriverProxyMountResponse
	)

	req.Name = name
	if err = pp.Call("VolumeDriver.Mount", req, &ret); err != nil {
		return
	}

	mountpoint = ret.Mountpoint

	if ret.Err != "" {
		err = errors.New(ret.Err)
	}

	return
}

type volumeDriverProxyUnmountRequest struct {
	Name string
}

type volumeDriverProxyUnmountResponse struct {
	Err string
}

func (pp *volumeDriverProxy) Unmount(name string) (err error) {
	var (
		req volumeDriverProxyUnmountRequest
		ret volumeDriverProxyUnmountResponse
	)

	req.Name = name
	if err = pp.Call("VolumeDriver.Unmount", req, &ret); err != nil {
		return
	}

	if ret.Err != "" {
		err = errors.New(ret.Err)
	}

	return
}

package graphdriver

import (
	"errors"
	"fmt"
	"io"
	"path/filepath"

	"github.com/docker/docker/pkg/archive"
	"github.com/docker/docker/pkg/containerfs"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/plugingetter"
	"github.com/docker/docker/pkg/plugins"
)

type graphDriverProxy struct {
	name string
	p    plugingetter.CompatPlugin
	caps Capabilities
}

type graphDriverRequest struct {
	ID         string            `json:",omitempty"`
	Parent     string            `json:",omitempty"`
	MountLabel string            `json:",omitempty"`
	StorageOpt map[string]string `json:",omitempty"`
}

type graphDriverResponse struct {
	Err          string            `json:",omitempty"`
	Dir          string            `json:",omitempty"`
	Exists       bool              `json:",omitempty"`
	Status       [][2]string       `json:",omitempty"`
	Changes      []archive.Change  `json:",omitempty"`
	Size         int64             `json:",omitempty"`
	Metadata     map[string]string `json:",omitempty"`
	Capabilities Capabilities      `json:",omitempty"`
}

type graphDriverInitRequest struct {
	Home    string
	Opts    []string        `json:"Opts"`
	UIDMaps []idtools.IDMap `json:"UIDMaps"`
	GIDMaps []idtools.IDMap `json:"GIDMaps"`
}

func (d *graphDriverProxy) Init(home string, opts []string, uidMaps, gidMaps []idtools.IDMap) error {
	if !d.p.IsV1() {
		if cp, ok := d.p.(plugingetter.CountedPlugin); ok {
			// always acquire here, it will be cleaned up on daemon shutdown
			cp.Acquire()
		}
	}
	args := &graphDriverInitRequest{
		Home:    home,
		Opts:    opts,
		UIDMaps: uidMaps,
		GIDMaps: gidMaps,
	}
	var ret graphDriverResponse
	if err := d.p.Client().Call("GraphDriver.Init", args, &ret); err != nil {
		return err
	}
	if ret.Err != "" {
		return errors.New(ret.Err)
	}
	caps, err := d.fetchCaps()
	if err != nil {
		return err
	}
	d.caps = caps
	return nil
}

func (d *graphDriverProxy) fetchCaps() (Capabilities, error) {
	args := &graphDriverRequest{}
	var ret graphDriverResponse
	if err := d.p.Client().Call("GraphDriver.Capabilities", args, &ret); err != nil {
		if !plugins.IsNotFound(err) {
			return Capabilities{}, err
		}
	}
	return ret.Capabilities, nil
}

func (d *graphDriverProxy) String() string {
	return d.name
}

func (d *graphDriverProxy) Capabilities() Capabilities {
	return d.caps
}

func (d *graphDriverProxy) CreateReadWrite(id, parent string, opts *CreateOpts) error {
	return d.create("GraphDriver.CreateReadWrite", id, parent, opts)
}

func (d *graphDriverProxy) Create(id, parent string, opts *CreateOpts) error {
	return d.create("GraphDriver.Create", id, parent, opts)
}

func (d *graphDriverProxy) create(method, id, parent string, opts *CreateOpts) error {
	args := &graphDriverRequest{
		ID:     id,
		Parent: parent,
	}
	if opts != nil {
		args.MountLabel = opts.MountLabel
		args.StorageOpt = opts.StorageOpt
	}
	var ret graphDriverResponse
	if err := d.p.Client().Call(method, args, &ret); err != nil {
		return err
	}
	if ret.Err != "" {
		return errors.New(ret.Err)
	}
	return nil
}

func (d *graphDriverProxy) Remove(id string) error {
	args := &graphDriverRequest{ID: id}
	var ret graphDriverResponse
	if err := d.p.Client().Call("GraphDriver.Remove", args, &ret); err != nil {
		return err
	}
	if ret.Err != "" {
		return errors.New(ret.Err)
	}
	return nil
}

func (d *graphDriverProxy) Get(id, mountLabel string) (containerfs.ContainerFS, error) {
	args := &graphDriverRequest{
		ID:         id,
		MountLabel: mountLabel,
	}
	var ret graphDriverResponse
	if err := d.p.Client().Call("GraphDriver.Get", args, &ret); err != nil {
		return nil, err
	}
	var err error
	if ret.Err != "" {
		err = errors.New(ret.Err)
	}
	return containerfs.NewLocalContainerFS(filepath.Join(d.p.BasePath(), ret.Dir)), err
}

func (d *graphDriverProxy) Put(id string) error {
	args := &graphDriverRequest{ID: id}
	var ret graphDriverResponse
	if err := d.p.Client().Call("GraphDriver.Put", args, &ret); err != nil {
		return err
	}
	if ret.Err != "" {
		return errors.New(ret.Err)
	}
	return nil
}

func (d *graphDriverProxy) Exists(id string) bool {
	args := &graphDriverRequest{ID: id}
	var ret graphDriverResponse
	if err := d.p.Client().Call("GraphDriver.Exists", args, &ret); err != nil {
		return false
	}
	return ret.Exists
}

func (d *graphDriverProxy) Status() [][2]string {
	args := &graphDriverRequest{}
	var ret graphDriverResponse
	if err := d.p.Client().Call("GraphDriver.Status", args, &ret); err != nil {
		return nil
	}
	return ret.Status
}

func (d *graphDriverProxy) GetMetadata(id string) (map[string]string, error) {
	args := &graphDriverRequest{
		ID: id,
	}
	var ret graphDriverResponse
	if err := d.p.Client().Call("GraphDriver.GetMetadata", args, &ret); err != nil {
		return nil, err
	}
	if ret.Err != "" {
		return nil, errors.New(ret.Err)
	}
	return ret.Metadata, nil
}

func (d *graphDriverProxy) Cleanup() error {
	if !d.p.IsV1() {
		if cp, ok := d.p.(plugingetter.CountedPlugin); ok {
			// always release
			defer cp.Release()
		}
	}

	args := &graphDriverRequest{}
	var ret graphDriverResponse
	if err := d.p.Client().Call("GraphDriver.Cleanup", args, &ret); err != nil {
		return nil
	}
	if ret.Err != "" {
		return errors.New(ret.Err)
	}
	return nil
}

func (d *graphDriverProxy) Diff(id, parent string) (io.ReadCloser, error) {
	args := &graphDriverRequest{
		ID:     id,
		Parent: parent,
	}
	body, err := d.p.Client().Stream("GraphDriver.Diff", args)
	if err != nil {
		return nil, err
	}
	return body, nil
}

func (d *graphDriverProxy) Changes(id, parent string) ([]archive.Change, error) {
	args := &graphDriverRequest{
		ID:     id,
		Parent: parent,
	}
	var ret graphDriverResponse
	if err := d.p.Client().Call("GraphDriver.Changes", args, &ret); err != nil {
		return nil, err
	}
	if ret.Err != "" {
		return nil, errors.New(ret.Err)
	}

	return ret.Changes, nil
}

func (d *graphDriverProxy) ApplyDiff(id, parent string, diff io.Reader) (int64, error) {
	var ret graphDriverResponse
	if err := d.p.Client().SendFile(fmt.Sprintf("GraphDriver.ApplyDiff?id=%s&parent=%s", id, parent), diff, &ret); err != nil {
		return -1, err
	}
	if ret.Err != "" {
		return -1, errors.New(ret.Err)
	}
	return ret.Size, nil
}

func (d *graphDriverProxy) DiffSize(id, parent string) (int64, error) {
	args := &graphDriverRequest{
		ID:     id,
		Parent: parent,
	}
	var ret graphDriverResponse
	if err := d.p.Client().Call("GraphDriver.DiffSize", args, &ret); err != nil {
		return -1, err
	}
	if ret.Err != "" {
		return -1, errors.New(ret.Err)
	}
	return ret.Size, nil
}

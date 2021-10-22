/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package export

import (
	"bytes"
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha512"
	"flag"
	"fmt"
	"hash"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/nfc"
	"github.com/vmware/govmomi/ovf"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type ovfx struct {
	*flags.VirtualMachineFlag

	dest   string
	name   string
	force  bool
	images bool
	sha    int

	mf bytes.Buffer
}

var sha = map[int]func() hash.Hash{
	1:   sha1.New,
	256: sha256.New,
	512: sha512.New,
}

func init() {
	cli.Register("export.ovf", &ovfx{})
}

func (cmd *ovfx) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	f.StringVar(&cmd.name, "name", "", "Specifies target name (defaults to source name)")
	f.BoolVar(&cmd.force, "f", false, "Overwrite existing")
	f.BoolVar(&cmd.images, "i", false, "Include image files (*.{iso,img})")
	f.IntVar(&cmd.sha, "sha", 0, "Generate manifest using SHA 1, 256, 512 or 0 to skip")
}

func (cmd *ovfx) Usage() string {
	return "DIR"
}

func (cmd *ovfx) Description() string {
	return `Export VM.

Examples:
  govc export.ovf -vm $vm DIR`
}

func (cmd *ovfx) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *ovfx) Run(ctx context.Context, f *flag.FlagSet) error {
	if f.NArg() != 1 {
		// TODO: output summary similar to ovftool's
		return flag.ErrHelp
	}

	vm, err := cmd.VirtualMachine()
	if err != nil {
		return err
	}

	if vm == nil {
		return flag.ErrHelp
	}

	if cmd.sha != 0 {
		if _, ok := sha[cmd.sha]; !ok {
			return fmt.Errorf("unknown hash: sha%d", cmd.sha)
		}
	}

	if cmd.name == "" {
		cmd.name = vm.Name()
	}

	cmd.dest = filepath.Join(f.Arg(0), cmd.name)

	target := filepath.Join(cmd.dest, cmd.name+".ovf")

	if !cmd.force {
		if _, err = os.Stat(target); err == nil {
			return fmt.Errorf("File already exists: %s", target)
		}
	}

	if err = os.MkdirAll(cmd.dest, 0755); err != nil {
		return err
	}

	lease, err := vm.Export(ctx)
	if err != nil {
		return err
	}

	info, err := lease.Wait(ctx, nil)
	if err != nil {
		return err
	}

	u := lease.StartUpdater(ctx, info)
	defer u.Done()

	cdp := types.OvfCreateDescriptorParams{
		Name: cmd.name,
	}

	for _, i := range info.Items {
		if !cmd.include(&i) {
			continue
		}

		if !strings.HasPrefix(i.Path, cmd.name) {
			i.Path = cmd.name + "-" + i.Path
		}

		err = cmd.Download(ctx, lease, i)
		if err != nil {
			return err
		}

		cdp.OvfFiles = append(cdp.OvfFiles, i.File())
	}

	if err = lease.Complete(ctx); err != nil {
		return err
	}

	m := ovf.NewManager(vm.Client())

	desc, err := m.CreateDescriptor(ctx, vm, cdp)
	if err != nil {
		return err
	}

	file, err := os.Create(target)
	if err != nil {
		return err
	}

	var w io.Writer = file
	h, ok := cmd.newHash()
	if ok {
		w = io.MultiWriter(file, h)
	}

	_, err = io.WriteString(w, desc.OvfDescriptor)
	if err != nil {
		return err
	}

	if err = file.Close(); err != nil {
		return err
	}

	if cmd.sha == 0 {
		return nil
	}

	cmd.addHash(filepath.Base(target), h)

	file, err = os.Create(filepath.Join(cmd.dest, cmd.name+".mf"))
	if err != nil {
		return err
	}

	_, err = io.Copy(file, &cmd.mf)
	if err != nil {
		return err
	}

	return file.Close()
}

func (cmd *ovfx) include(item *nfc.FileItem) bool {
	if cmd.images {
		return true
	}

	return filepath.Ext(item.Path) == ".vmdk"
}

func (cmd *ovfx) newHash() (hash.Hash, bool) {
	if h, ok := sha[cmd.sha]; ok {
		return h(), true
	}

	return nil, false
}

func (cmd *ovfx) addHash(p string, h hash.Hash) {
	_, _ = fmt.Fprintf(&cmd.mf, "SHA%d(%s)= %x\n", cmd.sha, p, h.Sum(nil))
}

func (cmd *ovfx) Download(ctx context.Context, lease *nfc.Lease, item nfc.FileItem) error {
	path := filepath.Join(cmd.dest, item.Path)

	logger := cmd.ProgressLogger(fmt.Sprintf("Downloading %s... ", item.Path))
	defer logger.Wait()

	opts := soap.Download{
		Progress: logger,
	}

	if h, ok := cmd.newHash(); ok {
		opts.Writer = h

		defer cmd.addHash(item.Path, h)
	}

	return lease.DownloadFile(ctx, path, item, opts)
}

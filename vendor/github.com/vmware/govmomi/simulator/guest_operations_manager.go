/*
Copyright (c) 2020 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"bufio"
	"fmt"
	"net/url"
	"strings"
	"syscall"
	"time"

	"github.com/vmware/govmomi/toolbox/process"
	"github.com/vmware/govmomi/toolbox/vix"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type GuestOperationsManager struct {
	mo.GuestOperationsManager
}

func (m *GuestOperationsManager) init(r *Registry) {
	fm := new(GuestFileManager)
	if m.FileManager == nil {
		m.FileManager = &types.ManagedObjectReference{
			Type:  "GuestFileManager",
			Value: "guestOperationsFileManager",
		}
	}
	fm.Self = *m.FileManager
	r.Put(fm)

	pm := new(GuestProcessManager)
	if m.ProcessManager == nil {
		m.ProcessManager = &types.ManagedObjectReference{
			Type:  "GuestProcessManager",
			Value: "guestOperationsProcessManager",
		}
	}
	pm.Self = *m.ProcessManager
	pm.Manager = process.NewManager()
	r.Put(pm)
}

type GuestFileManager struct {
	mo.GuestFileManager
}

func guestURL(ctx *Context, vm *VirtualMachine, path string) string {
	return (&url.URL{
		Scheme: ctx.svc.Listen.Scheme,
		Host:   "*", // See guest.FileManager.TransferURL
		Path:   guestPrefix + strings.TrimPrefix(path, "/"),
		RawQuery: url.Values{
			"id":    []string{vm.run.id},
			"token": []string{ctx.Session.Key},
		}.Encode(),
	}).String()
}

func (m *GuestFileManager) InitiateFileTransferToGuest(ctx *Context, req *types.InitiateFileTransferToGuest) soap.HasFault {
	body := new(methods.InitiateFileTransferToGuestBody)

	vm := ctx.Map.Get(req.Vm).(*VirtualMachine)
	err := vm.run.prepareGuestOperation(vm, req.Auth)
	if err != nil {
		body.Fault_ = Fault("", err)
		return body
	}

	body.Res = &types.InitiateFileTransferToGuestResponse{
		Returnval: guestURL(ctx, vm, req.GuestFilePath),
	}

	return body
}

func (m *GuestFileManager) InitiateFileTransferFromGuest(ctx *Context, req *types.InitiateFileTransferFromGuest) soap.HasFault {
	body := new(methods.InitiateFileTransferFromGuestBody)

	vm := ctx.Map.Get(req.Vm).(*VirtualMachine)
	err := vm.run.prepareGuestOperation(vm, req.Auth)
	if err != nil {
		body.Fault_ = Fault("", err)
		return body
	}

	body.Res = &types.InitiateFileTransferFromGuestResponse{
		Returnval: types.FileTransferInformation{
			Attributes: nil, // TODO
			Size:       0,   // TODO
			Url:        guestURL(ctx, vm, req.GuestFilePath),
		},
	}

	return body
}

type GuestProcessManager struct {
	mo.GuestProcessManager
	*process.Manager
}

func (m *GuestProcessManager) StartProgramInGuest(ctx *Context, req *types.StartProgramInGuest) soap.HasFault {
	body := new(methods.StartProgramInGuestBody)

	spec := req.Spec.(*types.GuestProgramSpec)
	auth := req.Auth.(*types.NamePasswordAuthentication)

	vm := ctx.Map.Get(req.Vm).(*VirtualMachine)

	fault := vm.run.prepareGuestOperation(vm, auth)
	if fault != nil {
		body.Fault_ = Fault("", fault)
	}

	args := []string{"exec"}

	if spec.WorkingDirectory != "" {
		args = append(args, "-w", spec.WorkingDirectory)
	}

	for _, e := range spec.EnvVariables {
		args = append(args, "-e", e)
	}

	args = append(args, vm.run.id, spec.ProgramPath, spec.Arguments)

	spec.ProgramPath = "docker"
	spec.Arguments = strings.Join(args, " ")

	start := &vix.StartProgramRequest{
		ProgramPath: spec.ProgramPath,
		Arguments:   spec.Arguments,
	}

	proc := process.New()
	proc.Owner = auth.Username

	pid, err := m.Start(start, proc)
	if err != nil {
		panic(err) // only happens if LookPath("docker") fails, which it should't at this point
	}

	body.Res = &types.StartProgramInGuestResponse{
		Returnval: pid,
	}

	return body
}

func (m *GuestProcessManager) ListProcessesInGuest(ctx *Context, req *types.ListProcessesInGuest) soap.HasFault {
	body := &methods.ListProcessesInGuestBody{
		Res: new(types.ListProcessesInGuestResponse),
	}

	procs := m.List(req.Pids)

	for _, proc := range procs {
		var end *time.Time
		if proc.EndTime != 0 {
			end = types.NewTime(time.Unix(proc.EndTime, 0))
		}

		body.Res.Returnval = append(body.Res.Returnval, types.GuestProcessInfo{
			Name:      proc.Name,
			Pid:       proc.Pid,
			Owner:     proc.Owner,
			CmdLine:   proc.Name + " " + proc.Args,
			StartTime: time.Unix(proc.StartTime, 0),
			EndTime:   end,
			ExitCode:  proc.ExitCode,
		})
	}

	return body
}

func (m *GuestProcessManager) TerminateProcessInGuest(ctx *Context, req *types.TerminateProcessInGuest) soap.HasFault {
	body := new(methods.TerminateProcessInGuestBody)

	if m.Kill(req.Pid) {
		body.Res = new(types.TerminateProcessInGuestResponse)
	} else {
		body.Fault_ = Fault("", &types.GuestProcessNotFound{Pid: req.Pid})
	}

	return body
}

func (m *GuestFileManager) mktemp(ctx *Context, req *types.CreateTemporaryFileInGuest, dir bool) (string, types.BaseMethodFault) {
	args := []string{"mktemp", fmt.Sprintf("--tmpdir=%s", req.DirectoryPath), req.Prefix + "vcsim-XXXXX" + req.Suffix}
	if dir {
		args = append(args, "-d")
	}

	vm := ctx.Map.Get(req.Vm).(*VirtualMachine)

	return vm.run.exec(ctx, vm, req.Auth, args)
}

func (m *GuestFileManager) CreateTemporaryFileInGuest(ctx *Context, req *types.CreateTemporaryFileInGuest) soap.HasFault {
	body := new(methods.CreateTemporaryFileInGuestBody)

	res, fault := m.mktemp(ctx, req, false)
	if fault != nil {
		body.Fault_ = Fault("", fault)
		return body
	}

	body.Res = &types.CreateTemporaryFileInGuestResponse{Returnval: res}

	return body
}

func (m *GuestFileManager) CreateTemporaryDirectoryInGuest(ctx *Context, req *types.CreateTemporaryDirectoryInGuest) soap.HasFault {
	body := new(methods.CreateTemporaryDirectoryInGuestBody)

	dir := types.CreateTemporaryFileInGuest(*req)
	res, fault := m.mktemp(ctx, &dir, true)
	if fault != nil {
		body.Fault_ = Fault("", fault)
		return body
	}

	body.Res = &types.CreateTemporaryDirectoryInGuestResponse{Returnval: res}

	return body
}

func listFiles(req *types.ListFilesInGuest) []string {
	args := []string{"find", req.FilePath}
	if req.MatchPattern != "" {
		args = append(args, "-name", req.MatchPattern)
	}
	return append(args, "-maxdepth", "1", "-exec", "stat", "-c", "%s %u %g %f %X %Y %n", "{}", "+")
}

func toFileInfo(s string) []types.GuestFileInfo {
	var res []types.GuestFileInfo

	scanner := bufio.NewScanner(strings.NewReader(s))

	for scanner.Scan() {
		var mode, atime, mtime int64
		attr := &types.GuestPosixFileAttributes{OwnerId: new(int32), GroupId: new(int32)}
		info := types.GuestFileInfo{Attributes: attr}

		_, err := fmt.Sscanf(scanner.Text(), "%d %d %d %x %d %d %s",
			&info.Size, attr.OwnerId, attr.GroupId, &mode, &atime, &mtime, &info.Path)
		if err != nil {
			panic(err)
		}

		attr.AccessTime = types.NewTime(time.Unix(atime, 0))
		attr.ModificationTime = types.NewTime(time.Unix(mtime, 0))
		attr.Permissions = mode & 0777

		switch mode & syscall.S_IFMT {
		case syscall.S_IFDIR:
			info.Type = string(types.GuestFileTypeDirectory)
		case syscall.S_IFLNK:
			info.Type = string(types.GuestFileTypeSymlink)
		default:
			info.Type = string(types.GuestFileTypeFile)
		}

		res = append(res, info)
	}

	return res
}

func (m *GuestFileManager) ListFilesInGuest(ctx *Context, req *types.ListFilesInGuest) soap.HasFault {
	body := new(methods.ListFilesInGuestBody)

	vm := ctx.Map.Get(req.Vm).(*VirtualMachine)

	if req.FilePath == "" {
		body.Fault_ = Fault("", new(types.InvalidArgument))
		return body
	}

	res, fault := vm.run.exec(ctx, vm, req.Auth, listFiles(req))
	if fault != nil {
		body.Fault_ = Fault("", fault)
		return body
	}

	body.Res = new(types.ListFilesInGuestResponse)
	body.Res.Returnval.Files = toFileInfo(res)

	return body
}

func (m *GuestFileManager) DeleteFileInGuest(ctx *Context, req *types.DeleteFileInGuest) soap.HasFault {
	body := new(methods.DeleteFileInGuestBody)

	args := []string{"rm", req.FilePath}

	vm := ctx.Map.Get(req.Vm).(*VirtualMachine)

	_, fault := vm.run.exec(ctx, vm, req.Auth, args)
	if fault != nil {
		body.Fault_ = Fault("", fault)
		return body
	}

	body.Res = new(types.DeleteFileInGuestResponse)

	return body
}

func (m *GuestFileManager) DeleteDirectoryInGuest(ctx *Context, req *types.DeleteDirectoryInGuest) soap.HasFault {
	body := new(methods.DeleteDirectoryInGuestBody)

	args := []string{"rmdir", req.DirectoryPath}
	if req.Recursive {
		args = []string{"rm", "-rf", req.DirectoryPath}
	}

	vm := ctx.Map.Get(req.Vm).(*VirtualMachine)

	_, fault := vm.run.exec(ctx, vm, req.Auth, args)
	if fault != nil {
		body.Fault_ = Fault("", fault)
		return body
	}

	body.Res = new(types.DeleteDirectoryInGuestResponse)

	return body
}

func (m *GuestFileManager) MakeDirectoryInGuest(ctx *Context, req *types.MakeDirectoryInGuest) soap.HasFault {
	body := new(methods.MakeDirectoryInGuestBody)

	args := []string{"mkdir", req.DirectoryPath}
	if req.CreateParentDirectories {
		args = []string{"mkdir", "-p", req.DirectoryPath}
	}

	vm := ctx.Map.Get(req.Vm).(*VirtualMachine)

	_, fault := vm.run.exec(ctx, vm, req.Auth, args)
	if fault != nil {
		body.Fault_ = Fault("", fault)
		return body
	}

	body.Res = new(types.MakeDirectoryInGuestResponse)

	return body
}

func (m *GuestFileManager) MoveFileInGuest(ctx *Context, req *types.MoveFileInGuest) soap.HasFault {
	body := new(methods.MoveFileInGuestBody)

	args := []string{"mv"}
	if !req.Overwrite {
		args = append(args, "-n")
	}
	args = append(args, req.SrcFilePath, req.DstFilePath)

	vm := ctx.Map.Get(req.Vm).(*VirtualMachine)

	_, fault := vm.run.exec(ctx, vm, req.Auth, args)
	if fault != nil {
		body.Fault_ = Fault("", fault)
		return body
	}

	body.Res = new(types.MoveFileInGuestResponse)

	return body
}

func (m *GuestFileManager) MoveDirectoryInGuest(ctx *Context, req *types.MoveDirectoryInGuest) soap.HasFault {
	body := new(methods.MoveDirectoryInGuestBody)

	args := []string{"mv", req.SrcDirectoryPath, req.DstDirectoryPath}

	vm := ctx.Map.Get(req.Vm).(*VirtualMachine)

	_, fault := vm.run.exec(ctx, vm, req.Auth, args)
	if fault != nil {
		body.Fault_ = Fault("", fault)
		return body
	}

	body.Res = new(types.MoveDirectoryInGuestResponse)

	return body
}

func (m *GuestFileManager) ChangeFileAttributesInGuest(ctx *Context, req *types.ChangeFileAttributesInGuest) soap.HasFault {
	body := new(methods.ChangeFileAttributesInGuestBody)

	vm := ctx.Map.Get(req.Vm).(*VirtualMachine)

	attr, ok := req.FileAttributes.(*types.GuestPosixFileAttributes)
	if !ok {
		body.Fault_ = Fault("", new(types.OperationNotSupportedByGuest))
		return body
	}

	if attr.Permissions != 0 {
		args := []string{"chmod", fmt.Sprintf("%#o", attr.Permissions), req.GuestFilePath}

		_, fault := vm.run.exec(ctx, vm, req.Auth, args)
		if fault != nil {
			body.Fault_ = Fault("", fault)
			return body
		}
	}

	change := []struct {
		cmd string
		id  *int32
	}{
		{"chown", attr.OwnerId},
		{"chgrp", attr.GroupId},
	}

	for _, c := range change {
		if c.id != nil {
			args := []string{c.cmd, fmt.Sprintf("%d", *c.id), req.GuestFilePath}

			_, fault := vm.run.exec(ctx, vm, req.Auth, args)
			if fault != nil {
				body.Fault_ = Fault("", fault)
				return body
			}
		}
	}

	body.Res = new(types.ChangeFileAttributesInGuestResponse)

	return body
}

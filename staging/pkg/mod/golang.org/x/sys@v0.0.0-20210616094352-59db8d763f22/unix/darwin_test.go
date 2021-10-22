// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build darwin && go1.12 && amd64
// +build darwin,go1.12,amd64

package unix

import (
	"os"
	"os/exec"
	"strings"
	"testing"
)

type darwinTest struct {
	name string
	f    uintptr
}

// TODO(khr): decide whether to keep this test enabled permanently or
// only temporarily.
func TestDarwinLoader(t *testing.T) {
	// Make sure the Darwin dynamic loader can actually resolve
	// all the system calls into libSystem.dylib. Unfortunately
	// there is no easy way to test this at compile time. So we
	// implement a crazy hack here, calling into the syscall
	// function with all its arguments set to junk, and see what
	// error we get. We are happy with any error (or none) except
	// an error from the dynamic loader.
	//
	// We have to run each test in a separate subprocess for fault isolation.
	//
	// Hopefully the junk args won't accidentally ask the system to do "rm -fr /".
	//
	// In an ideal world each syscall would have its own test, so this test
	// would be unnecessary. Unfortunately, we do not live in that world.
	for _, test := range darwinTests {
		// Call the test binary recursively, giving it a magic argument
		// (see init below) and the name of the test to run.
		cmd := exec.Command(os.Args[0], "testDarwinLoader", test.name)

		// Run subprocess, collect results. Note that we expect the subprocess
		// to fail somehow, so the error is irrelevant.
		out, _ := cmd.CombinedOutput()

		if strings.Contains(string(out), "dyld: Symbol not found:") {
			t.Errorf("can't resolve %s in libSystem.dylib", test.name)
		}
		if !strings.Contains(string(out), "success") {
			// Not really an error. Might be a syscall that never returns,
			// like exit, or one that segfaults, like gettimeofday.
			t.Logf("test never finished: %s: %s", test.name, string(out))
		}
	}
}

func init() {
	// The test binary execs itself with the "testDarwinLoader" argument.
	// Run the test specified by os.Args[2], then panic.
	if len(os.Args) >= 3 && os.Args[1] == "testDarwinLoader" {
		for _, test := range darwinTests {
			if test.name == os.Args[2] {
				syscall_syscall(test.f, ^uintptr(0), ^uintptr(0), ^uintptr(0))
			}
		}
		// Panic with a "success" label, so the parent process can check it.
		panic("success")
	}
}

// All the _trampoline functions in zsyscall_darwin_$ARCH.s
var darwinTests = [...]darwinTest{
	{"getgroups", libc_getgroups_trampoline_addr},
	{"setgroups", libc_setgroups_trampoline_addr},
	{"wait4", libc_wait4_trampoline_addr},
	{"accept", libc_accept_trampoline_addr},
	{"bind", libc_bind_trampoline_addr},
	{"connect", libc_connect_trampoline_addr},
	{"socket", libc_socket_trampoline_addr},
	{"getsockopt", libc_getsockopt_trampoline_addr},
	{"setsockopt", libc_setsockopt_trampoline_addr},
	{"getpeername", libc_getpeername_trampoline_addr},
	{"getsockname", libc_getsockname_trampoline_addr},
	{"shutdown", libc_shutdown_trampoline_addr},
	{"socketpair", libc_socketpair_trampoline_addr},
	{"recvfrom", libc_recvfrom_trampoline_addr},
	{"sendto", libc_sendto_trampoline_addr},
	{"recvmsg", libc_recvmsg_trampoline_addr},
	{"sendmsg", libc_sendmsg_trampoline_addr},
	{"kevent", libc_kevent_trampoline_addr},
	{"sysctl", libc_sysctl_trampoline_addr},
	{"utimes", libc_utimes_trampoline_addr},
	{"futimes", libc_futimes_trampoline_addr},
	{"fcntl", libc_fcntl_trampoline_addr},
	{"poll", libc_poll_trampoline_addr},
	{"madvise", libc_madvise_trampoline_addr},
	{"mlock", libc_mlock_trampoline_addr},
	{"mlockall", libc_mlockall_trampoline_addr},
	{"mprotect", libc_mprotect_trampoline_addr},
	{"msync", libc_msync_trampoline_addr},
	{"munlock", libc_munlock_trampoline_addr},
	{"munlockall", libc_munlockall_trampoline_addr},
	{"ptrace", libc_ptrace_trampoline_addr},
	{"pipe", libc_pipe_trampoline_addr},
	{"getxattr", libc_getxattr_trampoline_addr},
	{"fgetxattr", libc_fgetxattr_trampoline_addr},
	{"setxattr", libc_setxattr_trampoline_addr},
	{"fsetxattr", libc_fsetxattr_trampoline_addr},
	{"removexattr", libc_removexattr_trampoline_addr},
	{"fremovexattr", libc_fremovexattr_trampoline_addr},
	{"listxattr", libc_listxattr_trampoline_addr},
	{"flistxattr", libc_flistxattr_trampoline_addr},
	{"kill", libc_kill_trampoline_addr},
	{"ioctl", libc_ioctl_trampoline_addr},
	{"access", libc_access_trampoline_addr},
	{"adjtime", libc_adjtime_trampoline_addr},
	{"chdir", libc_chdir_trampoline_addr},
	{"chflags", libc_chflags_trampoline_addr},
	{"chmod", libc_chmod_trampoline_addr},
	{"chown", libc_chown_trampoline_addr},
	{"chroot", libc_chroot_trampoline_addr},
	{"close", libc_close_trampoline_addr},
	{"dup", libc_dup_trampoline_addr},
	{"dup2", libc_dup2_trampoline_addr},
	{"exchangedata", libc_exchangedata_trampoline_addr},
	{"exit", libc_exit_trampoline_addr},
	{"faccessat", libc_faccessat_trampoline_addr},
	{"fchdir", libc_fchdir_trampoline_addr},
	{"fchflags", libc_fchflags_trampoline_addr},
	{"fchmod", libc_fchmod_trampoline_addr},
	{"fchmodat", libc_fchmodat_trampoline_addr},
	{"fchown", libc_fchown_trampoline_addr},
	{"fchownat", libc_fchownat_trampoline_addr},
	{"flock", libc_flock_trampoline_addr},
	{"fpathconf", libc_fpathconf_trampoline_addr},
	{"fstat64", libc_fstat64_trampoline_addr},
	{"fstatat64", libc_fstatat64_trampoline_addr},
	{"fstatfs64", libc_fstatfs64_trampoline_addr},
	{"fsync", libc_fsync_trampoline_addr},
	{"ftruncate", libc_ftruncate_trampoline_addr},
	{"getdtablesize", libc_getdtablesize_trampoline_addr},
	{"getegid", libc_getegid_trampoline_addr},
	{"geteuid", libc_geteuid_trampoline_addr},
	{"getgid", libc_getgid_trampoline_addr},
	{"getpgid", libc_getpgid_trampoline_addr},
	{"getpgrp", libc_getpgrp_trampoline_addr},
	{"getpid", libc_getpid_trampoline_addr},
	{"getppid", libc_getppid_trampoline_addr},
	{"getpriority", libc_getpriority_trampoline_addr},
	{"getrlimit", libc_getrlimit_trampoline_addr},
	{"getrusage", libc_getrusage_trampoline_addr},
	{"getsid", libc_getsid_trampoline_addr},
	{"getuid", libc_getuid_trampoline_addr},
	{"issetugid", libc_issetugid_trampoline_addr},
	{"kqueue", libc_kqueue_trampoline_addr},
	{"lchown", libc_lchown_trampoline_addr},
	{"link", libc_link_trampoline_addr},
	{"linkat", libc_linkat_trampoline_addr},
	{"listen", libc_listen_trampoline_addr},
	{"lstat64", libc_lstat64_trampoline_addr},
	{"mkdir", libc_mkdir_trampoline_addr},
	{"mkdirat", libc_mkdirat_trampoline_addr},
	{"mkfifo", libc_mkfifo_trampoline_addr},
	{"mknod", libc_mknod_trampoline_addr},
	{"open", libc_open_trampoline_addr},
	{"openat", libc_openat_trampoline_addr},
	{"pathconf", libc_pathconf_trampoline_addr},
	{"pread", libc_pread_trampoline_addr},
	{"pwrite", libc_pwrite_trampoline_addr},
	{"read", libc_read_trampoline_addr},
	{"readlink", libc_readlink_trampoline_addr},
	{"readlinkat", libc_readlinkat_trampoline_addr},
	{"rename", libc_rename_trampoline_addr},
	{"renameat", libc_renameat_trampoline_addr},
	{"revoke", libc_revoke_trampoline_addr},
	{"rmdir", libc_rmdir_trampoline_addr},
	{"lseek", libc_lseek_trampoline_addr},
	{"select", libc_select_trampoline_addr},
	{"setegid", libc_setegid_trampoline_addr},
	{"seteuid", libc_seteuid_trampoline_addr},
	{"setgid", libc_setgid_trampoline_addr},
	{"setlogin", libc_setlogin_trampoline_addr},
	{"setpgid", libc_setpgid_trampoline_addr},
	{"setpriority", libc_setpriority_trampoline_addr},
	{"setprivexec", libc_setprivexec_trampoline_addr},
	{"setregid", libc_setregid_trampoline_addr},
	{"setreuid", libc_setreuid_trampoline_addr},
	{"setrlimit", libc_setrlimit_trampoline_addr},
	{"setsid", libc_setsid_trampoline_addr},
	{"settimeofday", libc_settimeofday_trampoline_addr},
	{"setuid", libc_setuid_trampoline_addr},
	{"stat64", libc_stat64_trampoline_addr},
	{"statfs64", libc_statfs64_trampoline_addr},
	{"symlink", libc_symlink_trampoline_addr},
	{"symlinkat", libc_symlinkat_trampoline_addr},
	{"sync", libc_sync_trampoline_addr},
	{"truncate", libc_truncate_trampoline_addr},
	{"umask", libc_umask_trampoline_addr},
	{"undelete", libc_undelete_trampoline_addr},
	{"unlink", libc_unlink_trampoline_addr},
	{"unlinkat", libc_unlinkat_trampoline_addr},
	{"unmount", libc_unmount_trampoline_addr},
	{"write", libc_write_trampoline_addr},
	{"mmap", libc_mmap_trampoline_addr},
	{"munmap", libc_munmap_trampoline_addr},
	{"gettimeofday", libc_gettimeofday_trampoline_addr},
	{"getfsstat64", libc_getfsstat64_trampoline_addr},
}

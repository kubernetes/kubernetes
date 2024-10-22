package userns

import (
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"syscall"

	"github.com/sirupsen/logrus"
	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/configs"
)

type Mapping struct {
	UIDMappings []configs.IDMap
	GIDMappings []configs.IDMap
}

func (m Mapping) toSys() (uids, gids []syscall.SysProcIDMap) {
	for _, uid := range m.UIDMappings {
		uids = append(uids, syscall.SysProcIDMap{
			ContainerID: int(uid.ContainerID),
			HostID:      int(uid.HostID),
			Size:        int(uid.Size),
		})
	}
	for _, gid := range m.GIDMappings {
		gids = append(gids, syscall.SysProcIDMap{
			ContainerID: int(gid.ContainerID),
			HostID:      int(gid.HostID),
			Size:        int(gid.Size),
		})
	}
	return
}

// id returns a unique identifier for this mapping, agnostic of the order of
// the uid and gid mappings (because the order doesn't matter to the kernel).
// The set of userns handles is indexed using this ID.
func (m Mapping) id() string {
	var uids, gids []string
	for _, idmap := range m.UIDMappings {
		uids = append(uids, fmt.Sprintf("%d:%d:%d", idmap.ContainerID, idmap.HostID, idmap.Size))
	}
	for _, idmap := range m.GIDMappings {
		gids = append(gids, fmt.Sprintf("%d:%d:%d", idmap.ContainerID, idmap.HostID, idmap.Size))
	}
	// We don't care about the sort order -- just sort them.
	sort.Strings(uids)
	sort.Strings(gids)
	return "uid=" + strings.Join(uids, ",") + ";gid=" + strings.Join(gids, ",")
}

type Handles struct {
	m    sync.Mutex
	maps map[string]*os.File
}

// Release all resources associated with this Handle. All existing files
// returned from Get() will continue to work even after calling Release(). The
// same Handles can be re-used after calling Release().
func (hs *Handles) Release() {
	hs.m.Lock()
	defer hs.m.Unlock()

	// Close the files for good measure, though GC will do that for us anyway.
	for _, file := range hs.maps {
		_ = file.Close()
	}
	hs.maps = nil
}

func spawnProc(req Mapping) (*os.Process, error) {
	// We need to spawn a subprocess with the requested mappings, which is
	// unfortunately quite expensive. The "safe" way of doing this is natively
	// with Go (and then spawning something like "sleep infinity"), but
	// execve() is a waste of cycles because we just need some process to have
	// the right mapping, we don't care what it's executing. The "unsafe"
	// option of doing a clone() behind the back of Go is probably okay in
	// theory as long as we just do kill(getpid(), SIGSTOP). However, if we
	// tell Go to put the new process into PTRACE_TRACEME mode, we can avoid
	// the exec and not have to faff around with the mappings.
	//
	// Note that Go's stdlib does not support newuidmap, but in the case of
	// id-mapped mounts, it seems incredibly unlikely that the user will be
	// requesting us to do a remapping as an unprivileged user with mappings
	// they have privileges over.
	logrus.Debugf("spawning dummy process for id-mapping %s", req.id())
	uidMappings, gidMappings := req.toSys()
	// We don't need to use /proc/thread-self here because the exe mm of a
	// thread-group is guaranteed to be the same for all threads by definition.
	// This lets us avoid having to do runtime.LockOSThread.
	return os.StartProcess("/proc/self/exe", []string{"runc", "--help"}, &os.ProcAttr{
		Sys: &syscall.SysProcAttr{
			Cloneflags:                 unix.CLONE_NEWUSER,
			UidMappings:                uidMappings,
			GidMappings:                gidMappings,
			GidMappingsEnableSetgroups: false,
			// Put the process into PTRACE_TRACEME mode to allow us to get the
			// userns without having a proper execve() target.
			Ptrace: true,
		},
	})
}

func dupFile(f *os.File) (*os.File, error) {
	newFd, err := unix.FcntlInt(f.Fd(), unix.F_DUPFD_CLOEXEC, 0)
	if err != nil {
		return nil, os.NewSyscallError("fcntl(F_DUPFD_CLOEXEC)", err)
	}
	return os.NewFile(uintptr(newFd), f.Name()), nil
}

// Get returns a handle to a /proc/$pid/ns/user nsfs file with the requested
// mapping. The processes spawned to produce userns nsfds are cached, so if
// equivalent user namespace mappings are requested, the same user namespace
// will be returned. The caller is responsible for closing the returned file
// descriptor.
func (hs *Handles) Get(req Mapping) (file *os.File, err error) {
	hs.m.Lock()
	defer hs.m.Unlock()

	if hs.maps == nil {
		hs.maps = make(map[string]*os.File)
	}

	file, ok := hs.maps[req.id()]
	if !ok {
		proc, err := spawnProc(req)
		if err != nil {
			return nil, fmt.Errorf("failed to spawn dummy process for map %s: %w", req.id(), err)
		}
		// Make sure we kill the helper process. We ignore errors because
		// there's not much we can do about them anyway, and ultimately
		defer func() {
			_ = proc.Kill()
			_, _ = proc.Wait()
		}()

		// Stash away a handle to the userns file. This is neater than keeping
		// the process alive, because Go's GC can handle files much better than
		// leaked processes, and having long-living useless processes seems
		// less than ideal.
		file, err = os.Open(fmt.Sprintf("/proc/%d/ns/user", proc.Pid))
		if err != nil {
			return nil, err
		}
		hs.maps[req.id()] = file
	}
	// Duplicate the file, to make sure the lifecycle of each *os.File we
	// return is independent.
	return dupFile(file)
}

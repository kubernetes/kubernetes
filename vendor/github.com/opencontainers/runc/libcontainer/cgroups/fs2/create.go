package fs2

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/opencontainers/runc/libcontainer/cgroups"
	"github.com/opencontainers/runc/libcontainer/configs"
)

func supportedControllers() (string, error) {
	return cgroups.ReadFile(UnifiedMountpoint, "/cgroup.controllers")
}

// needAnyControllers returns whether we enable some supported controllers or not,
// based on (1) controllers available and (2) resources that are being set.
// We don't check "pseudo" controllers such as
// "freezer" and "devices".
func needAnyControllers(r *configs.Resources) (bool, error) {
	if r == nil {
		return false, nil
	}

	// list of all available controllers
	content, err := supportedControllers()
	if err != nil {
		return false, err
	}
	avail := make(map[string]struct{})
	for _, ctr := range strings.Fields(content) {
		avail[ctr] = struct{}{}
	}

	// check whether the controller if available or not
	have := func(controller string) bool {
		_, ok := avail[controller]
		return ok
	}

	if isPidsSet(r) && have("pids") {
		return true, nil
	}
	if isMemorySet(r) && have("memory") {
		return true, nil
	}
	if isIoSet(r) && have("io") {
		return true, nil
	}
	if isCpuSet(r) && have("cpu") {
		return true, nil
	}
	if isCpusetSet(r) && have("cpuset") {
		return true, nil
	}
	if isHugeTlbSet(r) && have("hugetlb") {
		return true, nil
	}

	return false, nil
}

// containsDomainController returns whether the current config contains domain controller or not.
// Refer to: http://man7.org/linux/man-pages/man7/cgroups.7.html
// As at Linux 4.19, the following controllers are threaded: cpu, perf_event, and pids.
func containsDomainController(r *configs.Resources) bool {
	return isMemorySet(r) || isIoSet(r) || isCpuSet(r) || isHugeTlbSet(r)
}

// CreateCgroupPath creates cgroupv2 path, enabling all the supported controllers.
func CreateCgroupPath(path string, c *configs.Cgroup) (Err error) {
	if !strings.HasPrefix(path, UnifiedMountpoint) {
		return fmt.Errorf("invalid cgroup path %s", path)
	}

	content, err := supportedControllers()
	if err != nil {
		return err
	}

	const (
		cgTypeFile  = "cgroup.type"
		cgStCtlFile = "cgroup.subtree_control"
	)
	ctrs := strings.Fields(content)
	res := "+" + strings.Join(ctrs, " +")

	elements := strings.Split(path, "/")
	elements = elements[3:]
	current := "/sys/fs"
	for i, e := range elements {
		current = filepath.Join(current, e)
		if i > 0 {
			if err := os.Mkdir(current, 0o755); err != nil {
				if !os.IsExist(err) {
					return err
				}
			} else {
				// If the directory was created, be sure it is not left around on errors.
				current := current
				defer func() {
					if Err != nil {
						os.Remove(current)
					}
				}()
			}
			cgType, _ := cgroups.ReadFile(current, cgTypeFile)
			cgType = strings.TrimSpace(cgType)
			switch cgType {
			// If the cgroup is in an invalid mode (usually this means there's an internal
			// process in the cgroup tree, because we created a cgroup under an
			// already-populated-by-other-processes cgroup), then we have to error out if
			// the user requested controllers which are not thread-aware. However, if all
			// the controllers requested are thread-aware we can simply put the cgroup into
			// threaded mode.
			case "domain invalid":
				if containsDomainController(c.Resources) {
					return fmt.Errorf("cannot enter cgroupv2 %q with domain controllers -- it is in an invalid state", current)
				} else {
					// Not entirely correct (in theory we'd always want to be a domain --
					// since that means we're a properly delegated cgroup subtree) but in
					// this case there's not much we can do and it's better than giving an
					// error.
					_ = cgroups.WriteFile(current, cgTypeFile, "threaded")
				}
			// If the cgroup is in (threaded) or (domain threaded) mode, we can only use thread-aware controllers
			// (and you cannot usually take a cgroup out of threaded mode).
			case "domain threaded":
				fallthrough
			case "threaded":
				if containsDomainController(c.Resources) {
					return fmt.Errorf("cannot enter cgroupv2 %q with domain controllers -- it is in %s mode", current, cgType)
				}
			}
		}
		// enable all supported controllers
		if i < len(elements)-1 {
			if err := cgroups.WriteFile(current, cgStCtlFile, res); err != nil {
				// try write one by one
				allCtrs := strings.Split(res, " ")
				for _, ctr := range allCtrs {
					_ = cgroups.WriteFile(current, cgStCtlFile, ctr)
				}
			}
			// Some controllers might not be enabled when rootless or containerized,
			// but we don't catch the error here. (Caught in setXXX() functions.)
		}
	}

	return nil
}

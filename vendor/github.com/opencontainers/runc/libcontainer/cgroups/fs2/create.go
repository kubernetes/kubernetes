package fs2

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"

	"github.com/opencontainers/runc/libcontainer/configs"
)

func supportedControllers(cgroup *configs.Cgroup) ([]byte, error) {
	const file = UnifiedMountpoint + "/cgroup.controllers"
	return ioutil.ReadFile(file)
}

// needAnyControllers returns whether we enable some supported controllers or not,
// based on (1) controllers available and (2) resources that are being set.
// We don't check "pseudo" controllers such as
// "freezer" and "devices".
func needAnyControllers(cgroup *configs.Cgroup) (bool, error) {
	if cgroup == nil {
		return false, nil
	}

	// list of all available controllers
	content, err := supportedControllers(cgroup)
	if err != nil {
		return false, err
	}
	avail := make(map[string]struct{})
	for _, ctr := range strings.Fields(string(content)) {
		avail[ctr] = struct{}{}
	}

	// check whether the controller if available or not
	have := func(controller string) bool {
		_, ok := avail[controller]
		return ok
	}

	if isPidsSet(cgroup) && have("pids") {
		return true, nil
	}
	if isMemorySet(cgroup) && have("memory") {
		return true, nil
	}
	if isIoSet(cgroup) && have("io") {
		return true, nil
	}
	if isCpuSet(cgroup) && have("cpu") {
		return true, nil
	}
	if isCpusetSet(cgroup) && have("cpuset") {
		return true, nil
	}
	if isHugeTlbSet(cgroup) && have("hugetlb") {
		return true, nil
	}

	return false, nil
}

// containsDomainController returns whether the current config contains domain controller or not.
// Refer to: http://man7.org/linux/man-pages/man7/cgroups.7.html
// As at Linux 4.19, the following controllers are threaded: cpu, perf_event, and pids.
func containsDomainController(cg *configs.Cgroup) bool {
	return isMemorySet(cg) || isIoSet(cg) || isCpuSet(cg) || isHugeTlbSet(cg)
}

// CreateCgroupPath creates cgroupv2 path, enabling all the supported controllers.
func CreateCgroupPath(path string, c *configs.Cgroup) (Err error) {
	if !strings.HasPrefix(path, UnifiedMountpoint) {
		return fmt.Errorf("invalid cgroup path %s", path)
	}

	content, err := supportedControllers(c)
	if err != nil {
		return err
	}

	ctrs := bytes.Fields(content)
	res := append([]byte("+"), bytes.Join(ctrs, []byte(" +"))...)

	elements := strings.Split(path, "/")
	elements = elements[3:]
	current := "/sys/fs"
	for i, e := range elements {
		current = filepath.Join(current, e)
		if i > 0 {
			if err := os.Mkdir(current, 0755); err != nil {
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
			cgTypeFile := filepath.Join(current, "cgroup.type")
			cgType, _ := ioutil.ReadFile(cgTypeFile)
			switch strings.TrimSpace(string(cgType)) {
			// If the cgroup is in an invalid mode (usually this means there's an internal
			// process in the cgroup tree, because we created a cgroup under an
			// already-populated-by-other-processes cgroup), then we have to error out if
			// the user requested controllers which are not thread-aware. However, if all
			// the controllers requested are thread-aware we can simply put the cgroup into
			// threaded mode.
			case "domain invalid":
				if containsDomainController(c) {
					return fmt.Errorf("cannot enter cgroupv2 %q with domain controllers -- it is in an invalid state", current)
				} else {
					// Not entirely correct (in theory we'd always want to be a domain --
					// since that means we're a properly delegated cgroup subtree) but in
					// this case there's not much we can do and it's better than giving an
					// error.
					_ = ioutil.WriteFile(cgTypeFile, []byte("threaded"), 0644)
				}
			// If the cgroup is in (threaded) or (domain threaded) mode, we can only use thread-aware controllers
			// (and you cannot usually take a cgroup out of threaded mode).
			case "domain threaded":
				fallthrough
			case "threaded":
				if containsDomainController(c) {
					return fmt.Errorf("cannot enter cgroupv2 %q with domain controllers -- it is in %s mode", current, strings.TrimSpace(string(cgType)))
				}
			}
		}
		// enable all supported controllers
		if i < len(elements)-1 {
			file := filepath.Join(current, "cgroup.subtree_control")
			if err := ioutil.WriteFile(file, res, 0644); err != nil {
				// try write one by one
				allCtrs := bytes.Split(res, []byte(" "))
				for _, ctr := range allCtrs {
					_ = ioutil.WriteFile(file, ctr, 0644)
				}
			}
			// Some controllers might not be enabled when rootless or containerized,
			// but we don't catch the error here. (Caught in setXXX() functions.)
		}
	}

	return nil
}

package daemon

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"

	"github.com/docker/docker/api/types"
)

func (daemon *Daemon) ContainerTop(name string, psArgs string) (*types.ContainerProcessList, error) {
	if psArgs == "" {
		psArgs = "-ef"
	}

	container, err := daemon.Get(name)
	if err != nil {
		return nil, err
	}

	if !container.IsRunning() {
		return nil, fmt.Errorf("Container %s is not running", name)
	}

	pids, err := daemon.ExecutionDriver().GetPidsForContainer(container.ID)
	if err != nil {
		return nil, err
	}

	output, err := exec.Command("ps", strings.Split(psArgs, " ")...).Output()
	if err != nil {
		return nil, fmt.Errorf("Error running ps: %s", err)
	}

	procList := &types.ContainerProcessList{}

	lines := strings.Split(string(output), "\n")
	procList.Titles = strings.Fields(lines[0])

	pidIndex := -1
	for i, name := range procList.Titles {
		if name == "PID" {
			pidIndex = i
		}
	}
	if pidIndex == -1 {
		return nil, fmt.Errorf("Couldn't find PID field in ps output")
	}

	for _, line := range lines[1:] {
		if len(line) == 0 {
			continue
		}
		fields := strings.Fields(line)
		p, err := strconv.Atoi(fields[pidIndex])
		if err != nil {
			return nil, fmt.Errorf("Unexpected pid '%s': %s", fields[pidIndex], err)
		}

		for _, pid := range pids {
			if pid == p {
				// Make sure number of fields equals number of header titles
				// merging "overhanging" fields
				process := fields[:len(procList.Titles)-1]
				process = append(process, strings.Join(fields[len(procList.Titles)-1:], " "))
				procList.Processes = append(procList.Processes, process)
			}
		}
	}
	container.LogEvent("top")
	return procList, nil
}

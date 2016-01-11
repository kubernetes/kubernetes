package idtools

import (
	"fmt"
	"os/exec"
	"path/filepath"
	"strings"
	"syscall"
)

// add a user and/or group to Linux /etc/passwd, /etc/group using standard
// Linux distribution commands:
// adduser --uid <id> --shell /bin/login --no-create-home --disabled-login --ingroup <groupname> <username>
// useradd -M -u <id> -s /bin/nologin -N -g <groupname> <username>
// addgroup --gid <id> <groupname>
// groupadd -g <id> <groupname>

const baseUID int = 10000
const baseGID int = 10000
const idMAX int = 65534

var (
	userCommand  string
	groupCommand string

	cmdTemplates = map[string]string{
		"adduser":  "--uid %d --shell /bin/false --no-create-home --disabled-login --ingroup %s %s",
		"useradd":  "-M -u %d -s /bin/false -N -g %s %s",
		"addgroup": "--gid %d %s",
		"groupadd": "-g %d %s",
	}
)

func init() {
	// set up which commands are used for adding users/groups dependent on distro
	if _, err := resolveBinary("adduser"); err == nil {
		userCommand = "adduser"
	} else if _, err := resolveBinary("useradd"); err == nil {
		userCommand = "useradd"
	}
	if _, err := resolveBinary("addgroup"); err == nil {
		groupCommand = "addgroup"
	} else if _, err := resolveBinary("groupadd"); err == nil {
		groupCommand = "groupadd"
	}
}

func resolveBinary(binname string) (string, error) {
	binaryPath, err := exec.LookPath(binname)
	if err != nil {
		return "", err
	}
	resolvedPath, err := filepath.EvalSymlinks(binaryPath)
	if err != nil {
		return "", err
	}
	//only return no error if the final resolved binary basename
	//matches what was searched for
	if filepath.Base(resolvedPath) == binname {
		return resolvedPath, nil
	}
	return "", fmt.Errorf("Binary %q does not resolve to a binary of that name in $PATH (%q)", binname, resolvedPath)
}

// AddNamespaceRangesUser takes a name and finds an unused uid, gid pair
// and calls the appropriate helper function to add the group and then
// the user to the group in /etc/group and /etc/passwd respectively.
// This new user's /etc/sub{uid,gid} ranges will be used for user namespace
// mapping ranges in containers.
func AddNamespaceRangesUser(name string) (int, int, error) {
	// Find unused uid, gid pair
	uid, err := findUnusedUID(baseUID)
	if err != nil {
		return -1, -1, fmt.Errorf("Unable to find unused UID: %v", err)
	}
	gid, err := findUnusedGID(baseGID)
	if err != nil {
		return -1, -1, fmt.Errorf("Unable to find unused GID: %v", err)
	}

	// First add the group that we will use
	if err := addGroup(name, gid); err != nil {
		return -1, -1, fmt.Errorf("Error adding group %q: %v", name, err)
	}
	// Add the user as a member of the group
	if err := addUser(name, uid, name); err != nil {
		return -1, -1, fmt.Errorf("Error adding user %q: %v", name, err)
	}
	return uid, gid, nil
}

func addUser(userName string, uid int, groupName string) error {

	if userCommand == "" {
		return fmt.Errorf("Cannot add user; no useradd/adduser binary found")
	}
	args := fmt.Sprintf(cmdTemplates[userCommand], uid, groupName, userName)
	return execAddCmd(userCommand, args)
}

func addGroup(groupName string, gid int) error {

	if groupCommand == "" {
		return fmt.Errorf("Cannot add group; no groupadd/addgroup binary found")
	}
	args := fmt.Sprintf(cmdTemplates[groupCommand], gid, groupName)
	// only error out if the error isn't that the group already exists
	// if the group exists then our needs are already met
	if err := execAddCmd(groupCommand, args); err != nil && !strings.Contains(err.Error(), "already exists") {
		return err
	}
	return nil
}

func execAddCmd(cmd, args string) error {
	execCmd := exec.Command(cmd, strings.Split(args, " ")...)
	out, err := execCmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("Failed to add user/group with error: %v; output: %q", err, string(out))
	}
	return nil
}

func findUnusedUID(startUID int) (int, error) {
	return findUnused("passwd", startUID)
}

func findUnusedGID(startGID int) (int, error) {
	return findUnused("group", startGID)
}

func findUnused(file string, id int) (int, error) {
	for {
		cmdStr := fmt.Sprintf("cat /etc/%s | cut -d: -f3 | grep '^%d$'", file, id)
		cmd := exec.Command("sh", "-c", cmdStr)
		if err := cmd.Run(); err != nil {
			// if a non-zero return code occurs, then we know the ID was not found
			// and is usable
			if exiterr, ok := err.(*exec.ExitError); ok {
				// The program has exited with an exit code != 0
				if status, ok := exiterr.Sys().(syscall.WaitStatus); ok {
					if status.ExitStatus() == 1 {
						//no match, we can use this ID
						return id, nil
					}
				}
			}
			return -1, fmt.Errorf("Error looking in /etc/%s for unused ID: %v", file, err)
		}
		id++
		if id > idMAX {
			return -1, fmt.Errorf("Maximum id in %q reached with finding unused numeric ID", file)
		}
	}
}

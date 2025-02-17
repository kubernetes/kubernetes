//go:build windows
// +build windows

/*
Copyright 2023 The Kubernetes Authors.

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

package filesystem

import (
	"fmt"
	"net"
	"os"
	"path/filepath"
	"strings"
	"time"

	"k8s.io/klog/v2"

	"golang.org/x/sys/windows"
)

const (
	// Amount of time to wait between attempting to use a Unix domain socket.
	// As detailed in https://github.com/kubernetes/kubernetes/issues/104584
	// the first attempt will most likely fail, hence the need to retry
	socketDialRetryPeriod = 1 * time.Second
	// Overall timeout value to dial a Unix domain socket, including retries
	socketDialTimeout = 4 * time.Second
)

// IsUnixDomainSocket returns whether a given file is a AF_UNIX socket file
// Note that due to the retry logic inside, it could take up to 4 seconds
// to determine whether or not the file path supplied is a Unix domain socket
func IsUnixDomainSocket(filePath string) (bool, error) {
	// Note that querrying for the Reparse Points (https://docs.microsoft.com/en-us/windows/win32/fileio/reparse-points)
	// for the file (using FSCTL_GET_REPARSE_POINT) and checking for reparse tag: reparseTagSocket
	// does NOT work in 1809 if the socket file is created within a bind mounted directory by a container
	// and the FSCTL is issued in the host by the kubelet.

	// If the file does not exist, it cannot be a Unix domain socket.
	info, err := os.Stat(filePath)
	if err != nil {
		return false, fmt.Errorf("stat file %s failed: %w", filePath, err)
	}

	klog.V(6).InfoS("Function IsUnixDomainSocket starts", "filePath", filePath)
	// Use os.ModeSocket (introduced in Go 1.23 on Windows)
	if info.Mode()&os.ModeSocket != 0 {
		klog.V(6).InfoS("File identified as a Unix domain socket",
			"filePath", filePath)
		return true, nil
	}
	// Fallback: Single dial attempt
	klog.V(6).InfoS("ModeSocket check was inconclusive, attempting to dial the socket",
		"filePath", filePath)
	c, err := net.Dial("unix", filePath)
	if err == nil {
		c.Close()
		klog.V(6).InfoS("Socket dialed successfully", "filePath", filePath)
		return true, nil
	}
	klog.V(6).InfoS("File is not a Unix domain socket", "filePath", filePath, "err", err)
	return false, nil
}

// On Windows os.Mkdir all doesn't set any permissions so call the Chown function below to set
// permissions once the directory is created.
func MkdirAll(path string, perm os.FileMode) error {
	klog.V(6).InfoS("Function MkdirAll starts", "path", path, "perm", perm)
	if _, err := os.Stat(path); err == nil {
		// Path already exists: nothing to do.
		return nil
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("error checking path %s: %w", path, err)
	}

	err := os.MkdirAll(path, perm)
	if err != nil {
		return fmt.Errorf("error creating directory %s: %w", path, err)
	}

	err = Chmod(path, perm)
	if err != nil {
		return fmt.Errorf("error setting permissions for directory %s: %w", path, err)
	}

	return nil
}

const (
	// These aren't defined in the syscall package for Windows :(
	USER_READ      = 0x100
	USER_WRITE     = 0x80
	USER_EXECUTE   = 0x40
	GROUP_READ     = 0x20
	GROUP_WRITE    = 0x10
	GROUP_EXECUTE  = 0x8
	OTHERS_READ    = 0x4
	OTHERS_WRITE   = 0x2
	OTHERS_EXECUTE = 0x1
	USER_ALL       = USER_READ | USER_WRITE | USER_EXECUTE
	GROUP_ALL      = GROUP_READ | GROUP_WRITE | GROUP_EXECUTE
	OTHERS_ALL     = OTHERS_READ | OTHERS_WRITE | OTHERS_EXECUTE
)

// On Windows os.Chmod only sets the read-only flag on files, so we need to use Windows APIs to set the desired access on files / directories.
// The OWNER mode will set file permissions for the file owner SID, the GROUP mode will set file permissions for the file group SID,
// and the OTHERS mode will set file permissions for BUILTIN\Users.
// Please note that Windows containers can be run as one of two user accounts; ContainerUser or ContainerAdministrator.
// Containers run as ContainerAdministrator will inherit permissions from BUILTIN\Administrators,
// while containers run as ContainerUser will inherit permissions from BUILTIN\Users.
// Windows containers do not have the ability to run as a custom user account that is known to the host so the OTHERS group mode
// is used to grant / deny permissions of files on the hosts to the ContainerUser account.
func Chmod(path string, filemode os.FileMode) error {
	klog.V(6).InfoS("Function Chmod starts", "path", path, "filemode", filemode)
	// Get security descriptor for the file
	sd, err := windows.GetNamedSecurityInfo(
		path,
		windows.SE_FILE_OBJECT,
		windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION|windows.OWNER_SECURITY_INFORMATION|windows.GROUP_SECURITY_INFORMATION)
	if err != nil {
		return fmt.Errorf("Error getting security descriptor for file %s: %v", path, err)
	}

	// Get owner SID from the security descriptor for assigning USER permissions
	owner, _, err := sd.Owner()
	if err != nil {
		return fmt.Errorf("Error getting owner SID for file %s: %v", path, err)
	}
	ownerString := owner.String()

	// Get the group SID from the security descriptor for assigning GROUP permissions
	group, _, err := sd.Group()
	if err != nil {
		return fmt.Errorf("Error getting group SID for file %s: %v", path, err)
	}
	groupString := group.String()

	mask := uint32(windows.ACCESS_MASK(filemode))

	// Build a new Discretionary Access Control List (DACL) with the desired permissions using
	//the Security Descriptor Definition Language (SDDL) format.
	// https://learn.microsoft.com/windows/win32/secauthz/security-descriptor-definition-language
	// the DACL is a list of Access Control Entries (ACEs) where each ACE represents the permissions (Allow or Deny) for a specific SID.
	// Each ACE has the following format:
	//  (AceType;AceFlags;Rights;ObjectGuid;InheritObjectGuid;AccountSid)
	// We can leave ObjectGuid and InheritObjectGuid empty for our purposes.

	dacl := "D:"

	// build the owner ACE
	dacl += "(A;OICI;"
	if mask&USER_ALL == USER_ALL {
		dacl += "FA"
	} else {
		if mask&USER_READ == USER_READ {
			dacl += "FR"
		}
		if mask&USER_WRITE == USER_WRITE {
			dacl += "FW"
		}
		if mask&USER_EXECUTE == USER_EXECUTE {
			dacl += "FX"
		}
	}
	dacl += ";;;" + ownerString + ")"

	// Build the group ACE
	dacl += "(A;OICI;"
	if mask&GROUP_ALL == GROUP_ALL {
		dacl += "FA"
	} else {
		if mask&GROUP_READ == GROUP_READ {
			dacl += "FR"
		}
		if mask&GROUP_WRITE == GROUP_WRITE {
			dacl += "FW"
		}
		if mask&GROUP_EXECUTE == GROUP_EXECUTE {
			dacl += "FX"
		}
	}
	dacl += ";;;" + groupString + ")"

	// Build the others ACE
	dacl += "(A;OICI;"
	if mask&OTHERS_ALL == OTHERS_ALL {
		dacl += "FA"
	} else {
		if mask&OTHERS_READ == OTHERS_READ {
			dacl += "FR"
		}
		if mask&OTHERS_WRITE == OTHERS_WRITE {
			dacl += "FW"
		}
		if mask&OTHERS_EXECUTE == OTHERS_EXECUTE {
			dacl += "FX"
		}
	}
	dacl += ";;;BU)"

	klog.V(6).InfoS("Setting new DACL for path", "path", path, "dacl", dacl)

	// create a new security descriptor from the DACL string
	newSD, err := windows.SecurityDescriptorFromString(dacl)
	if err != nil {
		return fmt.Errorf("Error creating new security descriptor from DACL string: %v", err)
	}

	// get the DACL in binary format from the newly created security descriptor
	newDACL, _, err := newSD.DACL()
	if err != nil {
		return fmt.Errorf("Error getting DACL from new security descriptor: %v", err)
	}

	// Write the new security descriptor to the file
	return windows.SetNamedSecurityInfo(
		path,
		windows.SE_FILE_OBJECT,
		windows.DACL_SECURITY_INFORMATION|windows.PROTECTED_DACL_SECURITY_INFORMATION,
		nil, // owner SID
		nil, // group SID
		newDACL,
		nil) // SACL
}

// IsAbs returns whether the given path is absolute or not.
// On Windows, filepath.IsAbs will not return True for paths prefixed with a slash, even
// though they can be used as absolute paths (https://docs.microsoft.com/en-us/dotnet/standard/io/file-path-formats).
//
// WARN: It isn't safe to use this for API values which will propagate across systems (e.g. REST API values
// that get validated on Unix, persisted, then consumed by Windows, etc).
func IsAbs(path string) bool {
	return filepath.IsAbs(path) || strings.HasPrefix(path, `\`) || strings.HasPrefix(path, `/`)
}

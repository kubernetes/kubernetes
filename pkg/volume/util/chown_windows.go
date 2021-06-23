// +build windows

/*
Copyright 2021 The Kubernetes Authors.

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

package util

import (
	"github.com/Microsoft/go-winio"
	"golang.org/x/sys/windows"
	"k8s.io/klog/v2"
)

// chown implements the Windows version applying user ownership to files
// Remove the following later:
// Open the file
// Call GetSecurityInfo to get the existing ACL
// Create an ACE and call SetEntriesInAcl to create a new ACL with it added
// Call SetSecurityInfo to set the new ACL back onto the file
// If you're just looking to set the owner, then yes, I think you would just use SetNamedSecurityInfo.
// Set objectType to windows.SE_FILE_OBJECT, securityInformation to windows.OWNER_SECURITY_INFORMATION,
// owner to the SID of the owner (obtained using something like LookupSID, and group, dacl, and sacl to nil
func chown(name, userName string) error {
	handle, err := windows.Open(name, windows.O_RDWR, windows.S_IWRITE)
	if err != nil {
		klog.Errorf("error opening %s: %v", name, err)
		return err
	}
	defer windows.Close(handle)

	// Start debug
	sd, err := windows.GetSecurityInfo(handle, windows.SE_FILE_OBJECT, windows.OWNER_SECURITY_INFORMATION)
	if err != nil {
		klog.Errorf("error getting security info for %s: %v\n", name, err)
		return err
	}
	klog.Infof("SD: %v\n", *sd)

	owner, _, err := sd.Owner()
	if owner == nil {
		klog.Infoln("Owner is nil")
	} else {
		klog.Infof("Owner: %s\n", owner)
	}

	group, _, err := sd.Group()
	if group == nil {
		klog.Infoln("Group is nil")
	} else {
		klog.Infof("Group: %s\n", group)
	}
	// End debug

	// E0623 01:20:06.408417    3544 chown_windows.go:44] error looking up SID ContainerAdministrator: No mapping between account names and security IDs was done.
	sid, _, _, err := windows.LookupSID("", userName)
	if err != nil {
		klog.Errorf("error looking up SID %s: %v\n", userName, err)
		return err
	}
	klog.Infof("SID: %s\n", sid)

	if err := winio.RunWithPrivilege(winio.SeRestorePrivilege, func() error {
		if err := windows.SetSecurityInfo(handle, windows.SE_FILE_OBJECT, windows.OWNER_SECURITY_INFORMATION, sid,
			nil, nil, nil); err != nil {
			klog.Errorf("error setting security info: %v\n", err)
			return err
		}
		return nil
	}); err != nil {
		return err
	}

	return nil
}

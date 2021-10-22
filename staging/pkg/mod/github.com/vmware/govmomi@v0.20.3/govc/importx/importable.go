/*
Copyright (c) 2014 VMware, Inc. All Rights Reserved.

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

package importx

import (
	"fmt"
	"path"
)

type importable struct {
	localPath  string
	remotePath string
}

func (i importable) Ext() string {
	return path.Ext(i.localPath)
}

func (i importable) Base() string {
	return path.Base(i.localPath)
}

func (i importable) BaseClean() string {
	b := i.Base()
	e := i.Ext()
	return b[:len(b)-len(e)]
}

func (i importable) RemoteSrcVMDK() string {
	file := fmt.Sprintf("%s-src.vmdk", i.BaseClean())
	return i.toRemotePath(file)
}

func (i importable) RemoteDstVMDK() string {
	file := fmt.Sprintf("%s.vmdk", i.BaseClean())
	return i.toRemotePath(file)
}

func (i importable) toRemotePath(p string) string {
	if i.remotePath == "" {
		return p
	}

	return path.Join(i.remotePath, p)
}

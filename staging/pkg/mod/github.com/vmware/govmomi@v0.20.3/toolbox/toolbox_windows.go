/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package toolbox

import (
	"fmt"
	"os"
)

func fileExtendedInfoFormat(dir string, info os.FileInfo) string {
	const format = "<fxi>" +
		"<Name>%s</Name>" +
		"<ft>%d</ft>" +
		"<fs>%d</fs>" +
		"<mt>%d</mt>" +
		"<ct>%d</ct>" +
		"<at>%d</at>" +
		"</fxi>"

	props := 0
	size := info.Size()
	mtime := info.ModTime().Unix()
	ctime := 0
	atime := 0

	return fmt.Sprintf(format, info.Name(), props, size, mtime, ctime, atime)
}

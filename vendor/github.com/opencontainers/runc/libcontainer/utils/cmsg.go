// +build linux

package utils

/*
 * Copyright 2016 SUSE LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
#include <errno.h>
#include <stdlib.h>
#include "cmsg.h"
*/
import "C"

import (
	"os"
	"unsafe"
)

// RecvFd waits for a file descriptor to be sent over the given AF_UNIX
// socket. The file name of the remote file descriptor will be recreated
// locally (it is sent as non-auxiliary data in the same payload).
func RecvFd(socket *os.File) (*os.File, error) {
	file, err := C.recvfd(C.int(socket.Fd()))
	if err != nil {
		return nil, err
	}
	defer C.free(unsafe.Pointer(file.name))
	return os.NewFile(uintptr(file.fd), C.GoString(file.name)), nil
}

// SendFd sends a file descriptor over the given AF_UNIX socket. In
// addition, the file.Name() of the given file will also be sent as
// non-auxiliary data in the same payload (allowing to send contextual
// information for a file descriptor).
func SendFd(socket, file *os.File) error {
	var cfile C.struct_file_t
	cfile.fd = C.int(file.Fd())
	cfile.name = C.CString(file.Name())
	defer C.free(unsafe.Pointer(cfile.name))

	_, err := C.sendfd(C.int(socket.Fd()), cfile)
	return err
}

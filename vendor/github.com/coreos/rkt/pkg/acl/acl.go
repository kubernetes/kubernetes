// Copyright 2016 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Copyright 2015 Joseph Naegele
//
// Portions of this code are derived from go-acl by Joseph Naegele
// (https://github.com/naegeldjd/go-acl) which is under an MIT license that can
// be found in the accompanying LICENSE.MIT file
//

//+build linux

// Package acl is a wrapper over libacl that dlopens it instead of being linked
// to it. The code is based on go-acl by Joseph Naegele
// (https://github.com/naegelejd/go-acl).
package acl

// #cgo LDFLAGS: -ldl
// #include <stdlib.h>
// #include <dlfcn.h>
// #include <stdio.h>
// #include <sys/acl.h>
// #include <sys/types.h>
// #include <unistd.h>
//
// acl_t
// my_acl_from_text(void *f, const char *acl)
// {
//   acl_t (*acl_from_text)(const char *);
//
//   acl_from_text = f;
//   return acl_from_text(acl);
// }
//
// int
// my_acl_set_file(void *f, const char *path_p, acl_type_t type, acl_t acl)
// {
//   int (*acl_set_file)(const char *, acl_type_t, acl_t);
//
//   acl_set_file = f;
//   return acl_set_file(path_p, type, acl);
// }
//
// int
// my_acl_free(void *f, acl_t acl)
// {
//   int (*acl_free)(acl_t);
//
//   acl_free = f;
//   return acl_free(acl);
// }
//
// int
// my_acl_valid(void *f, acl_t acl)
// {
//   int (*acl_valid)(acl_t);
//
//   acl_valid = f;
//   return acl_valid(acl);
// }
//
// int
// my_acl_create_entry(void *f, acl_t *acl_p, acl_entry_t *entry_p)
// {
//   int (*acl_create_entry)(acl_t *, acl_entry_t *);
//
//   acl_create_entry = f;
//   return acl_create_entry(acl_p, entry_p);
// }
//
// int
// my_acl_set_tag_type(void *f, acl_entry_t entry_d, acl_tag_t tag_type)
// {
//   int (*acl_set_tag_type)(acl_entry_t, acl_tag_t);
//
//   acl_set_tag_type = f;
//   return acl_set_tag_type(entry_d, tag_type);
// }
//
// int
// my_acl_get_permset(void *f, acl_entry_t entry_d, acl_permset_t *permset_p)
// {
//   int (*acl_get_permset)(acl_entry_t, acl_permset_t *);
//
//   acl_get_permset = f;
//   return acl_get_permset(entry_d, permset_p);
// }
//
// int
// my_acl_add_perm(void *f, acl_permset_t permset_d, acl_perm_t perm)
// {
//   int (*acl_add_perm)(acl_permset_t, acl_perm_t);
//
//   acl_add_perm = f;
//   return acl_add_perm(permset_d, perm);
// }
import "C"

import (
	"errors"
	"fmt"
	"os"
	"unsafe"

	"github.com/hashicorp/errwrap"
)

const (
	otherExec = 1 << iota
	otherWrite
	otherRead
	groupExec
	groupWrite
	groupRead
	userExec
	userWrite
	userRead
)

const (
	TagUserObj  Tag = C.ACL_USER_OBJ
	TagUser         = C.ACL_USER
	TagGroupObj     = C.ACL_GROUP_OBJ
	TagGroup        = C.ACL_GROUP
	TagMask         = C.ACL_MASK
	TagOther        = C.ACL_OTHER
)

const (
	PermRead    Perm = C.ACL_READ
	PermWrite        = C.ACL_WRITE
	PermExecute      = C.ACL_EXECUTE
)

type ACL struct {
	lib *libHandle
	a   C.acl_t
}

// Entry is an entry in an ACL.
type Entry struct {
	a *ACL
	e C.acl_entry_t
}

// Permset is a set of permissions.
type Permset struct {
	a *ACL
	p C.acl_permset_t
}

// Perm represents a permission in the e_perm ACL field
type Perm int

// Tag represents an ACL e_tag entry
type Tag int

var ErrSoNotFound = errors.New("unable to open a handle to libacl")

type libHandle struct {
	handle  unsafe.Pointer
	libname string
}

func getHandle() (*libHandle, error) {
	for _, name := range []string{
		"libacl.so.1",
		"libacl.so",
	} {
		libname := C.CString(name)
		defer C.free(unsafe.Pointer(libname))
		handle := C.dlopen(libname, C.RTLD_LAZY)
		if handle != nil {
			h := &libHandle{
				handle:  handle,
				libname: name,
			}
			return h, nil
		}
	}
	return nil, ErrSoNotFound
}

func getSymbolPointer(handle unsafe.Pointer, symbol string) (unsafe.Pointer, error) {
	sym := C.CString(symbol)
	defer C.free(unsafe.Pointer(sym))

	C.dlerror()
	p := C.dlsym(handle, sym)
	e := C.dlerror()
	if e != nil {
		return nil, errwrap.Wrap(fmt.Errorf("error resolving symbol %q", symbol), errors.New(C.GoString(e)))
	}

	return p, nil
}

func (h *libHandle) close() error {
	C.dlerror()
	C.dlclose(h.handle)
	e := C.dlerror()
	if e != nil {
		return errwrap.Wrap(fmt.Errorf("error closing %v", h.libname), errors.New(C.GoString(e)))
	}
	return nil
}

// InitACL dlopens libacl and returns an ACL object if successful.
func InitACL() (*ACL, error) {
	h, err := getHandle()
	if err != nil {
		return nil, err
	}

	return &ACL{lib: h}, nil
}

// ParseACL parses a string representation of an ACL.
func (a *ACL) ParseACL(acl string) error {
	acl_from_text, err := getSymbolPointer(a.lib.handle, "acl_from_text")
	if err != nil {
		return err
	}
	cacl := C.CString(acl)
	defer C.free(unsafe.Pointer(cacl))

	retACL, err := C.my_acl_from_text(acl_from_text, cacl)
	if retACL == nil {
		return errwrap.Wrap(errors.New("error calling acl_from_text"), err)
	}

	a.a = retACL

	return nil
}

// Free frees libacl's internal structures and closes libacl.
func (a *ACL) Free() error {
	acl_free, err := getSymbolPointer(a.lib.handle, "acl_free")
	if err != nil {
		return err
	}

	ret, err := C.my_acl_free(acl_free, a.a)
	if ret < 0 {
		return errwrap.Wrap(errors.New("error calling acl_free"), err)
	}

	return a.lib.close()
}

// SetFileACLDefault sets the "default" ACL for path.
func (a *ACL) SetFileACLDefault(path string) error {
	acl_set_file, err := getSymbolPointer(a.lib.handle, "acl_set_file")
	if err != nil {
		return err
	}

	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))

	ret, err := C.my_acl_set_file(acl_set_file, cpath, C.ACL_TYPE_DEFAULT, a.a)
	if ret < 0 {
		return errwrap.Wrap(errors.New("error calling acl_set_file"), err)
	}

	return nil
}

// Valid checks whether the ACL is valid.
func (a *ACL) Valid() error {
	acl_valid, err := getSymbolPointer(a.lib.handle, "acl_valid")
	if err != nil {
		return err
	}

	ret, err := C.my_acl_valid(acl_valid, a.a)
	if ret < 0 {
		return errwrap.Wrap(errors.New("invalid acl"), err)
	}
	return nil
}

// AddBaseEntries adds the base ACL entries from the file permissions.
func (a *ACL) AddBaseEntries(path string) error {
	fi, err := os.Lstat(path)
	if err != nil {
		return err
	}
	mode := fi.Mode().Perm()
	var r, w, x bool

	// set USER_OBJ entry
	r = mode&userRead == userRead
	w = mode&userWrite == userWrite
	x = mode&userExec == userExec
	if err := a.addBaseEntryFromMode(TagUserObj, r, w, x); err != nil {
		return err
	}

	// set GROUP_OBJ entry
	r = mode&groupRead == groupRead
	w = mode&groupWrite == groupWrite
	x = mode&groupExec == groupExec
	if err := a.addBaseEntryFromMode(TagGroupObj, r, w, x); err != nil {
		return err
	}

	// set OTHER entry
	r = mode&otherRead == otherRead
	w = mode&otherWrite == otherWrite
	x = mode&otherExec == otherExec
	if err := a.addBaseEntryFromMode(TagOther, r, w, x); err != nil {
		return err
	}

	return nil
}

func (a *ACL) createEntry() (*Entry, error) {
	acl_create_entry, err := getSymbolPointer(a.lib.handle, "acl_create_entry")
	if err != nil {
		return nil, err
	}

	var e C.acl_entry_t

	rv, err := C.my_acl_create_entry(acl_create_entry, &a.a, &e)
	if rv < 0 {
		return nil, errwrap.Wrap(errors.New("unable to create entry"), err)
	}
	return &Entry{a, e}, nil
}

func (a *ACL) addBaseEntryFromMode(tag Tag, read, write, execute bool) error {
	e, err := a.createEntry()
	if err != nil {
		return err
	}
	if err = e.setTag(tag); err != nil {
		return err
	}
	p, err := e.getPermset(a)
	if err != nil {
		return err
	}
	if err := p.addPermsFromMode(read, write, execute); err != nil {
		return err
	}
	return nil
}

func (entry *Entry) getPermset(a *ACL) (*Permset, error) {
	acl_get_permset, err := getSymbolPointer(a.lib.handle, "acl_get_permset")
	if err != nil {
		return nil, err
	}

	var ps C.acl_permset_t
	rv, err := C.my_acl_get_permset(acl_get_permset, entry.e, &ps)
	if rv < 0 {
		return nil, errwrap.Wrap(errors.New("unable to get permset"), err)
	}
	return &Permset{a, ps}, nil
}

func (entry *Entry) setTag(t Tag) error {
	acl_set_tag_type, err := getSymbolPointer(entry.a.lib.handle, "acl_set_tag_type")
	if err != nil {
		return err
	}

	rv, err := C.my_acl_set_tag_type(acl_set_tag_type, entry.e, C.acl_tag_t(t))
	if rv < 0 {
		return errwrap.Wrap(errors.New("unable to set tag"), err)
	}

	return nil
}

func (pset *Permset) addPerm(perm Perm) error {
	acl_add_perm, err := getSymbolPointer(pset.a.lib.handle, "acl_add_perm")
	if err != nil {
		return err
	}

	rv, err := C.my_acl_add_perm(acl_add_perm, pset.p, C.acl_perm_t(perm))
	if rv < 0 {
		return errwrap.Wrap(errors.New("unable to add perm to permset"), err)
	}
	return nil
}

func (p *Permset) addPermsFromMode(read, write, execute bool) error {
	if read {
		if err := p.addPerm(PermRead); err != nil {
			return err
		}
	}
	if write {
		if err := p.addPerm(PermWrite); err != nil {
			return err
		}
	}
	if execute {
		if err := p.addPerm(PermExecute); err != nil {
			return err
		}
	}
	return nil
}

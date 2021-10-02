package regstate

import (
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"reflect"
	"syscall"

	"golang.org/x/sys/windows"
	"golang.org/x/sys/windows/registry"
)

//go:generate go run $GOROOT/src/syscall/mksyscall_windows.go -output zsyscall_windows.go regstate.go

//sys	regCreateKeyEx(key syscall.Handle, subkey *uint16, reserved uint32, class *uint16, options uint32, desired uint32, sa *syscall.SecurityAttributes, result *syscall.Handle, disposition *uint32) (regerrno error) = advapi32.RegCreateKeyExW

const (
	_REG_OPTION_VOLATILE = 1

	_REG_OPENED_EXISTING_KEY = 2
)

type Key struct {
	registry.Key
	Name string
}

var localMachine = &Key{registry.LOCAL_MACHINE, "HKEY_LOCAL_MACHINE"}
var localUser = &Key{registry.CURRENT_USER, "HKEY_CURRENT_USER"}

var rootPath = `SOFTWARE\Microsoft\runhcs`

type NotFoundError struct {
	Id string
}

func (err *NotFoundError) Error() string {
	return fmt.Sprintf("ID '%s' was not found", err.Id)
}

func IsNotFoundError(err error) bool {
	_, ok := err.(*NotFoundError)
	return ok
}

type NoStateError struct {
	ID  string
	Key string
}

func (err *NoStateError) Error() string {
	return fmt.Sprintf("state '%s' is not present for ID '%s'", err.Key, err.ID)
}

func createVolatileKey(k *Key, path string, access uint32) (newk *Key, openedExisting bool, err error) {
	var (
		h syscall.Handle
		d uint32
	)
	fullpath := filepath.Join(k.Name, path)
	pathPtr, _ := windows.UTF16PtrFromString(path)
	err = regCreateKeyEx(syscall.Handle(k.Key), pathPtr, 0, nil, _REG_OPTION_VOLATILE, access, nil, &h, &d)
	if err != nil {
		return nil, false, &os.PathError{Op: "RegCreateKeyEx", Path: fullpath, Err: err}
	}
	return &Key{registry.Key(h), fullpath}, d == _REG_OPENED_EXISTING_KEY, nil
}

func hive(perUser bool) *Key {
	r := localMachine
	if perUser {
		r = localUser
	}
	return r
}

func Open(root string, perUser bool) (*Key, error) {
	k, _, err := createVolatileKey(hive(perUser), rootPath, registry.ALL_ACCESS)
	if err != nil {
		return nil, err
	}
	defer k.Close()

	k2, _, err := createVolatileKey(k, url.PathEscape(root), registry.ALL_ACCESS)
	if err != nil {
		return nil, err
	}
	return k2, nil
}

func RemoveAll(root string, perUser bool) error {
	k, err := hive(perUser).open(rootPath)
	if err != nil {
		return err
	}
	defer k.Close()
	r, err := k.open(url.PathEscape(root))
	if err != nil {
		return err
	}
	defer r.Close()
	ids, err := r.Enumerate()
	if err != nil {
		return err
	}
	for _, id := range ids {
		err = r.Remove(id)
		if err != nil {
			return err
		}
	}
	r.Close()
	return k.Remove(root)
}

func (k *Key) Close() error {
	err := k.Key.Close()
	k.Key = 0
	return err
}

func (k *Key) Enumerate() ([]string, error) {
	escapedIDs, err := k.ReadSubKeyNames(0)
	if err != nil {
		return nil, err
	}
	var ids []string
	for _, e := range escapedIDs {
		id, err := url.PathUnescape(e)
		if err == nil {
			ids = append(ids, id)
		}
	}
	return ids, nil
}

func (k *Key) open(name string) (*Key, error) {
	fullpath := filepath.Join(k.Name, name)
	nk, err := registry.OpenKey(k.Key, name, registry.ALL_ACCESS)
	if err != nil {
		return nil, &os.PathError{Op: "RegOpenKey", Path: fullpath, Err: err}
	}
	return &Key{nk, fullpath}, nil
}

func (k *Key) openid(id string) (*Key, error) {
	escaped := url.PathEscape(id)
	fullpath := filepath.Join(k.Name, escaped)
	nk, err := k.open(escaped)
	if perr, ok := err.(*os.PathError); ok && perr.Err == syscall.ERROR_FILE_NOT_FOUND {
		return nil, &NotFoundError{id}
	}
	if err != nil {
		return nil, &os.PathError{Op: "RegOpenKey", Path: fullpath, Err: err}
	}
	return nk, nil
}

func (k *Key) Remove(id string) error {
	escaped := url.PathEscape(id)
	err := registry.DeleteKey(k.Key, escaped)
	if err != nil {
		if err == syscall.ERROR_FILE_NOT_FOUND {
			return &NotFoundError{id}
		}
		return &os.PathError{Op: "RegDeleteKey", Path: filepath.Join(k.Name, escaped), Err: err}
	}
	return nil
}

func (k *Key) set(id string, create bool, key string, state interface{}) error {
	var sk *Key
	var err error
	if create {
		var existing bool
		eid := url.PathEscape(id)
		sk, existing, err = createVolatileKey(k, eid, registry.ALL_ACCESS)
		if err != nil {
			return err
		}
		defer sk.Close()
		if existing {
			sk.Close()
			return fmt.Errorf("container %s already exists", id)
		}
	} else {
		sk, err = k.openid(id)
		if err != nil {
			return err
		}
		defer sk.Close()
	}
	switch reflect.TypeOf(state).Kind() {
	case reflect.Bool:
		v := uint32(0)
		if state.(bool) {
			v = 1
		}
		err = sk.SetDWordValue(key, v)
	case reflect.Int:
		err = sk.SetQWordValue(key, uint64(state.(int)))
	case reflect.String:
		err = sk.SetStringValue(key, state.(string))
	default:
		var js []byte
		js, err = json.Marshal(state)
		if err != nil {
			return err
		}
		err = sk.SetBinaryValue(key, js)
	}
	if err != nil {
		if err == syscall.ERROR_FILE_NOT_FOUND {
			return &NoStateError{id, key}
		}
		return &os.PathError{Op: "RegSetValueEx", Path: sk.Name + ":" + key, Err: err}
	}
	return nil
}

func (k *Key) Create(id, key string, state interface{}) error {
	return k.set(id, true, key, state)
}

func (k *Key) Set(id, key string, state interface{}) error {
	return k.set(id, false, key, state)
}

func (k *Key) Clear(id, key string) error {
	sk, err := k.openid(id)
	if err != nil {
		return err
	}
	defer sk.Close()
	err = sk.DeleteValue(key)
	if err != nil {
		if err == syscall.ERROR_FILE_NOT_FOUND {
			return &NoStateError{id, key}
		}
		return &os.PathError{Op: "RegDeleteValue", Path: sk.Name + ":" + key, Err: err}
	}
	return nil
}

func (k *Key) Get(id, key string, state interface{}) error {
	sk, err := k.openid(id)
	if err != nil {
		return err
	}
	defer sk.Close()

	var js []byte
	switch reflect.TypeOf(state).Elem().Kind() {
	case reflect.Bool:
		var v uint64
		v, _, err = sk.GetIntegerValue(key)
		if err == nil {
			*state.(*bool) = v != 0
		}
	case reflect.Int:
		var v uint64
		v, _, err = sk.GetIntegerValue(key)
		if err == nil {
			*state.(*int) = int(v)
		}
	case reflect.String:
		var v string
		v, _, err = sk.GetStringValue(key)
		if err == nil {
			*state.(*string) = string(v)
		}
	default:
		js, _, err = sk.GetBinaryValue(key)
	}
	if err != nil {
		if err == syscall.ERROR_FILE_NOT_FOUND {
			return &NoStateError{id, key}
		}
		return &os.PathError{Op: "RegQueryValueEx", Path: sk.Name + ":" + key, Err: err}
	}
	if js != nil {
		err = json.Unmarshal(js, state)
	}
	return err
}

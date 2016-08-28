package gofig

import (
	"bytes"
	"strings"
	"unicode"

	log "github.com/Sirupsen/logrus"
	"github.com/akutz/goof"
)

// Registration is used to register configuration information with the gofig
// package.
type Registration struct {
	name string
	yaml string
	keys []*regKey
}

// KeyType is a config registration key type.
type KeyType int

const (
	// String is a key with a string value
	String KeyType = iota

	// Int is a key with an integer value
	Int

	// Bool is a key with a boolean value
	Bool

	// SecureString is a key with a string value that is not included when the
	// configuration is marshaled to JSON.
	SecureString
)

type regKey struct {
	keyType    KeyType
	defVal     interface{}
	short      string
	desc       string
	keyName    string
	flagName   string
	envVarName string
}

// NewRegistration creates a new registration with the given name.
func NewRegistration(name string) *Registration {
	return &Registration{
		name: name,
		keys: []*regKey{},
	}
}

// Yaml sets the registration's default yaml configuration.
func (r *Registration) Yaml(y string) {
	r.yaml = y
}

// Key adds a key to the registration.
//
// The first vararg argument is the yaml name of the key, using a '.' as
// the nested separator. If the second two arguments are omitted they will be
// generated from the first argument. The second argument is the explicit name
// of the flag bound to this key. The third argument is the explicit name of
// the environment variable bound to thie key.
func (r *Registration) Key(
	keyType KeyType,
	short string,
	defVal interface{},
	description string,
	keys ...interface{}) {

	lk := len(keys)
	if lk == 0 {
		panic(goof.New("keys is empty"))
	}

	rk := &regKey{
		keyType: keyType,
		short:   short,
		desc:    description,
		defVal:  defVal,
		keyName: toString(keys[0]),
	}

	if keyType == SecureString {
		secureKey(rk)
	}

	if lk < 2 {
		kp := strings.Split(rk.keyName, ".")
		for x, s := range kp {
			if x == 0 {
				var buff []byte
				b := bytes.NewBuffer(buff)
				for y, r := range s {
					if y == 0 {
						b.WriteRune(unicode.ToLower(r))
					} else {
						b.WriteRune(r)
					}
				}
				kp[x] = b.String()
			} else {
				kp[x] = strings.Title(s)
			}
		}
		rk.flagName = strings.Join(kp, "")
	} else {
		rk.flagName = toString(keys[1])
	}

	if lk < 3 {
		kp := strings.Split(rk.keyName, ".")
		for x, s := range kp {
			kp[x] = strings.ToUpper(s)
		}
		rk.envVarName = strings.Join(kp, "_")
	} else {
		rk.envVarName = toString(keys[2])
	}

	r.keys = append(r.keys, rk)
}

func secureKey(k *regKey) {
	secureKeysRWL.Lock()
	defer secureKeysRWL.Unlock()
	kn := strings.ToLower(k.keyName)
	if LogSecureKey {
		log.WithField("keyName", kn).Debug("securing key")
	}
	secureKeys[kn] = k
}

package devices

import (
	"fmt"
	"os"
	"strconv"
)

const (
	Wildcard = -1
)

type Device struct {
	Rule

	// Path to the device.
	Path string `json:"path"`

	// FileMode permission bits for the device.
	FileMode os.FileMode `json:"file_mode"`

	// Uid of the device.
	Uid uint32 `json:"uid"`

	// Gid of the device.
	Gid uint32 `json:"gid"`
}

// Permissions is a cgroupv1-style string to represent device access. It
// has to be a string for backward compatibility reasons, hence why it has
// methods to do set operations.
type Permissions string

const (
	deviceRead uint = (1 << iota)
	deviceWrite
	deviceMknod
)

func (p Permissions) toSet() uint {
	var set uint
	for _, perm := range p {
		switch perm {
		case 'r':
			set |= deviceRead
		case 'w':
			set |= deviceWrite
		case 'm':
			set |= deviceMknod
		}
	}
	return set
}

func fromSet(set uint) Permissions {
	var perm string
	if set&deviceRead == deviceRead {
		perm += "r"
	}
	if set&deviceWrite == deviceWrite {
		perm += "w"
	}
	if set&deviceMknod == deviceMknod {
		perm += "m"
	}
	return Permissions(perm)
}

// Union returns the union of the two sets of Permissions.
func (p Permissions) Union(o Permissions) Permissions {
	lhs := p.toSet()
	rhs := o.toSet()
	return fromSet(lhs | rhs)
}

// Difference returns the set difference of the two sets of Permissions.
// In set notation, A.Difference(B) gives you A\B.
func (p Permissions) Difference(o Permissions) Permissions {
	lhs := p.toSet()
	rhs := o.toSet()
	return fromSet(lhs &^ rhs)
}

// Intersection computes the intersection of the two sets of Permissions.
func (p Permissions) Intersection(o Permissions) Permissions {
	lhs := p.toSet()
	rhs := o.toSet()
	return fromSet(lhs & rhs)
}

// IsEmpty returns whether the set of permissions in a Permissions is
// empty.
func (p Permissions) IsEmpty() bool {
	return p == Permissions("")
}

// IsValid returns whether the set of permissions is a subset of valid
// permissions (namely, {r,w,m}).
func (p Permissions) IsValid() bool {
	return p == fromSet(p.toSet())
}

type Type rune

const (
	WildcardDevice Type = 'a'
	BlockDevice    Type = 'b'
	CharDevice     Type = 'c' // or 'u'
	FifoDevice     Type = 'p'
)

func (t Type) IsValid() bool {
	switch t {
	case WildcardDevice, BlockDevice, CharDevice, FifoDevice:
		return true
	default:
		return false
	}
}

func (t Type) CanMknod() bool {
	switch t {
	case BlockDevice, CharDevice, FifoDevice:
		return true
	default:
		return false
	}
}

func (t Type) CanCgroup() bool {
	switch t {
	case WildcardDevice, BlockDevice, CharDevice:
		return true
	default:
		return false
	}
}

type Rule struct {
	// Type of device ('c' for char, 'b' for block). If set to 'a', this rule
	// acts as a wildcard and all fields other than Allow are ignored.
	Type Type `json:"type"`

	// Major is the device's major number.
	Major int64 `json:"major"`

	// Minor is the device's minor number.
	Minor int64 `json:"minor"`

	// Permissions is the set of permissions that this rule applies to (in the
	// cgroupv1 format -- any combination of "rwm").
	Permissions Permissions `json:"permissions"`

	// Allow specifies whether this rule is allowed.
	Allow bool `json:"allow"`
}

func (d *Rule) CgroupString() string {
	var (
		major = strconv.FormatInt(d.Major, 10)
		minor = strconv.FormatInt(d.Minor, 10)
	)
	if d.Major == Wildcard {
		major = "*"
	}
	if d.Minor == Wildcard {
		minor = "*"
	}
	return fmt.Sprintf("%c %s:%s %s", d.Type, major, minor, d.Permissions)
}

func (d *Rule) Mkdev() (uint64, error) {
	return mkDev(d)
}

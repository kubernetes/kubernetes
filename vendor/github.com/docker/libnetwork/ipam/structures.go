package ipam

import (
	"encoding/json"
	"fmt"
	"net"
	"strings"
	"sync"

	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/ipamapi"
	"github.com/docker/libnetwork/types"
)

// SubnetKey is the pointer to the configured pools in each address space
type SubnetKey struct {
	AddressSpace string
	Subnet       string
	ChildSubnet  string
}

// PoolData contains the configured pool data
type PoolData struct {
	ParentKey SubnetKey
	Pool      *net.IPNet
	Range     *AddressRange `json:",omitempty"`
	RefCount  int
}

// addrSpace contains the pool configurations for the address space
type addrSpace struct {
	subnets  map[SubnetKey]*PoolData
	dbIndex  uint64
	dbExists bool
	id       string
	scope    string
	ds       datastore.DataStore
	alloc    *Allocator
	sync.Mutex
}

// AddressRange specifies first and last ip ordinal which
// identifies a range in a pool of addresses
type AddressRange struct {
	Sub        *net.IPNet
	Start, End uint64
}

// String returns the string form of the AddressRange object
func (r *AddressRange) String() string {
	return fmt.Sprintf("Sub: %s, range [%d, %d]", r.Sub, r.Start, r.End)
}

// MarshalJSON returns the JSON encoding of the Range object
func (r *AddressRange) MarshalJSON() ([]byte, error) {
	m := map[string]interface{}{
		"Sub":   r.Sub.String(),
		"Start": r.Start,
		"End":   r.End,
	}
	return json.Marshal(m)
}

// UnmarshalJSON decodes data into the Range object
func (r *AddressRange) UnmarshalJSON(data []byte) error {
	m := map[string]interface{}{}
	err := json.Unmarshal(data, &m)
	if err != nil {
		return err
	}
	if r.Sub, err = types.ParseCIDR(m["Sub"].(string)); err != nil {
		return err
	}
	r.Start = uint64(m["Start"].(float64))
	r.End = uint64(m["End"].(float64))
	return nil
}

// String returns the string form of the SubnetKey object
func (s *SubnetKey) String() string {
	k := fmt.Sprintf("%s/%s", s.AddressSpace, s.Subnet)
	if s.ChildSubnet != "" {
		k = fmt.Sprintf("%s/%s", k, s.ChildSubnet)
	}
	return k
}

// FromString populates the SubnetKey object reading it from string
func (s *SubnetKey) FromString(str string) error {
	if str == "" || !strings.Contains(str, "/") {
		return types.BadRequestErrorf("invalid string form for subnetkey: %s", str)
	}

	p := strings.Split(str, "/")
	if len(p) != 3 && len(p) != 5 {
		return types.BadRequestErrorf("invalid string form for subnetkey: %s", str)
	}
	s.AddressSpace = p[0]
	s.Subnet = fmt.Sprintf("%s/%s", p[1], p[2])
	if len(p) == 5 {
		s.ChildSubnet = fmt.Sprintf("%s/%s", p[3], p[4])
	}

	return nil
}

// String returns the string form of the PoolData object
func (p *PoolData) String() string {
	return fmt.Sprintf("ParentKey: %s, Pool: %s, Range: %s, RefCount: %d",
		p.ParentKey.String(), p.Pool.String(), p.Range, p.RefCount)
}

// MarshalJSON returns the JSON encoding of the PoolData object
func (p *PoolData) MarshalJSON() ([]byte, error) {
	m := map[string]interface{}{
		"ParentKey": p.ParentKey,
		"RefCount":  p.RefCount,
	}
	if p.Pool != nil {
		m["Pool"] = p.Pool.String()
	}
	if p.Range != nil {
		m["Range"] = p.Range
	}
	return json.Marshal(m)
}

// UnmarshalJSON decodes data into the PoolData object
func (p *PoolData) UnmarshalJSON(data []byte) error {
	var (
		err error
		t   struct {
			ParentKey SubnetKey
			Pool      string
			Range     *AddressRange `json:",omitempty"`
			RefCount  int
		}
	)

	if err = json.Unmarshal(data, &t); err != nil {
		return err
	}

	p.ParentKey = t.ParentKey
	p.Range = t.Range
	p.RefCount = t.RefCount
	if t.Pool != "" {
		if p.Pool, err = types.ParseCIDR(t.Pool); err != nil {
			return err
		}
	}

	return nil
}

// MarshalJSON returns the JSON encoding of the addrSpace object
func (aSpace *addrSpace) MarshalJSON() ([]byte, error) {
	aSpace.Lock()
	defer aSpace.Unlock()

	m := map[string]interface{}{
		"Scope": string(aSpace.scope),
	}

	if aSpace.subnets != nil {
		s := map[string]*PoolData{}
		for k, v := range aSpace.subnets {
			s[k.String()] = v
		}
		m["Subnets"] = s
	}

	return json.Marshal(m)
}

// UnmarshalJSON decodes data into the addrSpace object
func (aSpace *addrSpace) UnmarshalJSON(data []byte) error {
	aSpace.Lock()
	defer aSpace.Unlock()

	m := map[string]interface{}{}
	err := json.Unmarshal(data, &m)
	if err != nil {
		return err
	}

	aSpace.scope = datastore.LocalScope
	s := m["Scope"].(string)
	if s == string(datastore.GlobalScope) {
		aSpace.scope = datastore.GlobalScope
	}

	if v, ok := m["Subnets"]; ok {
		sb, _ := json.Marshal(v)
		var s map[string]*PoolData
		err := json.Unmarshal(sb, &s)
		if err != nil {
			return err
		}
		for ks, v := range s {
			k := SubnetKey{}
			k.FromString(ks)
			aSpace.subnets[k] = v
		}
	}

	return nil
}

// CopyTo deep copies the pool data to the destination pooldata
func (p *PoolData) CopyTo(dstP *PoolData) error {
	dstP.ParentKey = p.ParentKey
	dstP.Pool = types.GetIPNetCopy(p.Pool)

	if p.Range != nil {
		dstP.Range = &AddressRange{}
		dstP.Range.Sub = types.GetIPNetCopy(p.Range.Sub)
		dstP.Range.Start = p.Range.Start
		dstP.Range.End = p.Range.End
	}

	dstP.RefCount = p.RefCount
	return nil
}

func (aSpace *addrSpace) CopyTo(o datastore.KVObject) error {
	aSpace.Lock()
	defer aSpace.Unlock()

	dstAspace := o.(*addrSpace)

	dstAspace.id = aSpace.id
	dstAspace.ds = aSpace.ds
	dstAspace.alloc = aSpace.alloc
	dstAspace.scope = aSpace.scope
	dstAspace.dbIndex = aSpace.dbIndex
	dstAspace.dbExists = aSpace.dbExists

	dstAspace.subnets = make(map[SubnetKey]*PoolData)
	for k, v := range aSpace.subnets {
		dstAspace.subnets[k] = &PoolData{}
		v.CopyTo(dstAspace.subnets[k])
	}

	return nil
}

func (aSpace *addrSpace) New() datastore.KVObject {
	aSpace.Lock()
	defer aSpace.Unlock()

	return &addrSpace{
		id:    aSpace.id,
		ds:    aSpace.ds,
		alloc: aSpace.alloc,
		scope: aSpace.scope,
	}
}

func (aSpace *addrSpace) updatePoolDBOnAdd(k SubnetKey, nw *net.IPNet, ipr *AddressRange, pdf bool) (func() error, error) {
	aSpace.Lock()
	defer aSpace.Unlock()

	// Check if already allocated
	if p, ok := aSpace.subnets[k]; ok {
		if pdf {
			return nil, types.InternalMaskableErrorf("predefined pool %s is already reserved", nw)
		}
		aSpace.incRefCount(p, 1)
		return func() error { return nil }, nil
	}

	// If master pool, check for overlap
	if ipr == nil {
		if aSpace.contains(k.AddressSpace, nw) {
			return nil, ipamapi.ErrPoolOverlap
		}
		// This is a new master pool, add it along with corresponding bitmask
		aSpace.subnets[k] = &PoolData{Pool: nw, RefCount: 1}
		return func() error { return aSpace.alloc.insertBitMask(k, nw) }, nil
	}

	// This is a new non-master pool
	p := &PoolData{
		ParentKey: SubnetKey{AddressSpace: k.AddressSpace, Subnet: k.Subnet},
		Pool:      nw,
		Range:     ipr,
		RefCount:  1,
	}
	aSpace.subnets[k] = p

	// Look for parent pool
	pp, ok := aSpace.subnets[p.ParentKey]
	if ok {
		aSpace.incRefCount(pp, 1)
		return func() error { return nil }, nil
	}

	// Parent pool does not exist, add it along with corresponding bitmask
	aSpace.subnets[p.ParentKey] = &PoolData{Pool: nw, RefCount: 1}
	return func() error { return aSpace.alloc.insertBitMask(p.ParentKey, nw) }, nil
}

func (aSpace *addrSpace) updatePoolDBOnRemoval(k SubnetKey) (func() error, error) {
	aSpace.Lock()
	defer aSpace.Unlock()

	p, ok := aSpace.subnets[k]
	if !ok {
		return nil, ipamapi.ErrBadPool
	}

	aSpace.incRefCount(p, -1)

	c := p
	for ok {
		if c.RefCount == 0 {
			delete(aSpace.subnets, k)
			if c.Range == nil {
				return func() error {
					bm, err := aSpace.alloc.retrieveBitmask(k, c.Pool)
					if err != nil {
						return types.InternalErrorf("could not find bitmask in datastore for pool %s removal: %v", k.String(), err)
					}
					return bm.Destroy()
				}, nil
			}
		}
		k = c.ParentKey
		c, ok = aSpace.subnets[k]
	}

	return func() error { return nil }, nil
}

func (aSpace *addrSpace) incRefCount(p *PoolData, delta int) {
	c := p
	ok := true
	for ok {
		c.RefCount += delta
		c, ok = aSpace.subnets[c.ParentKey]
	}
}

// Checks whether the passed subnet is a superset or subset of any of the subset in this config db
func (aSpace *addrSpace) contains(space string, nw *net.IPNet) bool {
	for k, v := range aSpace.subnets {
		if space == k.AddressSpace && k.ChildSubnet == "" {
			if nw.Contains(v.Pool.IP) || v.Pool.Contains(nw.IP) {
				return true
			}
		}
	}
	return false
}

func (aSpace *addrSpace) store() datastore.DataStore {
	aSpace.Lock()
	defer aSpace.Unlock()

	return aSpace.ds
}

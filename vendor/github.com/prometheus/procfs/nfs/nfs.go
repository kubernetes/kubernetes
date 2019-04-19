// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package nfs implements parsing of /proc/net/rpc/nfsd.
// Fields are documented in https://www.svennd.be/nfsd-stats-explained-procnetrpcnfsd/
package nfs

// ReplyCache models the "rc" line.
type ReplyCache struct {
	Hits    uint64
	Misses  uint64
	NoCache uint64
}

// FileHandles models the "fh" line.
type FileHandles struct {
	Stale        uint64
	TotalLookups uint64
	AnonLookups  uint64
	DirNoCache   uint64
	NoDirNoCache uint64
}

// InputOutput models the "io" line.
type InputOutput struct {
	Read  uint64
	Write uint64
}

// Threads models the "th" line.
type Threads struct {
	Threads uint64
	FullCnt uint64
}

// ReadAheadCache models the "ra" line.
type ReadAheadCache struct {
	CacheSize      uint64
	CacheHistogram []uint64
	NotFound       uint64
}

// Network models the "net" line.
type Network struct {
	NetCount   uint64
	UDPCount   uint64
	TCPCount   uint64
	TCPConnect uint64
}

// ClientRPC models the nfs "rpc" line.
type ClientRPC struct {
	RPCCount        uint64
	Retransmissions uint64
	AuthRefreshes   uint64
}

// ServerRPC models the nfsd "rpc" line.
type ServerRPC struct {
	RPCCount uint64
	BadCnt   uint64
	BadFmt   uint64
	BadAuth  uint64
	BadcInt  uint64
}

// V2Stats models the "proc2" line.
type V2Stats struct {
	Null     uint64
	GetAttr  uint64
	SetAttr  uint64
	Root     uint64
	Lookup   uint64
	ReadLink uint64
	Read     uint64
	WrCache  uint64
	Write    uint64
	Create   uint64
	Remove   uint64
	Rename   uint64
	Link     uint64
	SymLink  uint64
	MkDir    uint64
	RmDir    uint64
	ReadDir  uint64
	FsStat   uint64
}

// V3Stats models the "proc3" line.
type V3Stats struct {
	Null        uint64
	GetAttr     uint64
	SetAttr     uint64
	Lookup      uint64
	Access      uint64
	ReadLink    uint64
	Read        uint64
	Write       uint64
	Create      uint64
	MkDir       uint64
	SymLink     uint64
	MkNod       uint64
	Remove      uint64
	RmDir       uint64
	Rename      uint64
	Link        uint64
	ReadDir     uint64
	ReadDirPlus uint64
	FsStat      uint64
	FsInfo      uint64
	PathConf    uint64
	Commit      uint64
}

// ClientV4Stats models the nfs "proc4" line.
type ClientV4Stats struct {
	Null               uint64
	Read               uint64
	Write              uint64
	Commit             uint64
	Open               uint64
	OpenConfirm        uint64
	OpenNoattr         uint64
	OpenDowngrade      uint64
	Close              uint64
	Setattr            uint64
	FsInfo             uint64
	Renew              uint64
	SetClientID        uint64
	SetClientIDConfirm uint64
	Lock               uint64
	Lockt              uint64
	Locku              uint64
	Access             uint64
	Getattr            uint64
	Lookup             uint64
	LookupRoot         uint64
	Remove             uint64
	Rename             uint64
	Link               uint64
	Symlink            uint64
	Create             uint64
	Pathconf           uint64
	StatFs             uint64
	ReadLink           uint64
	ReadDir            uint64
	ServerCaps         uint64
	DelegReturn        uint64
	GetACL             uint64
	SetACL             uint64
	FsLocations        uint64
	ReleaseLockowner   uint64
	Secinfo            uint64
	FsidPresent        uint64
	ExchangeID         uint64
	CreateSession      uint64
	DestroySession     uint64
	Sequence           uint64
	GetLeaseTime       uint64
	ReclaimComplete    uint64
	LayoutGet          uint64
	GetDeviceInfo      uint64
	LayoutCommit       uint64
	LayoutReturn       uint64
	SecinfoNoName      uint64
	TestStateID        uint64
	FreeStateID        uint64
	GetDeviceList      uint64
	BindConnToSession  uint64
	DestroyClientID    uint64
	Seek               uint64
	Allocate           uint64
	DeAllocate         uint64
	LayoutStats        uint64
	Clone              uint64
}

// ServerV4Stats models the nfsd "proc4" line.
type ServerV4Stats struct {
	Null     uint64
	Compound uint64
}

// V4Ops models the "proc4ops" line: NFSv4 operations
// Variable list, see:
// v4.0 https://tools.ietf.org/html/rfc3010 (38 operations)
// v4.1 https://tools.ietf.org/html/rfc5661 (58 operations)
// v4.2 https://tools.ietf.org/html/draft-ietf-nfsv4-minorversion2-41 (71 operations)
type V4Ops struct {
	//Values       uint64 // Variable depending on v4.x sub-version. TODO: Will this always at least include the fields in this struct?
	Op0Unused    uint64
	Op1Unused    uint64
	Op2Future    uint64
	Access       uint64
	Close        uint64
	Commit       uint64
	Create       uint64
	DelegPurge   uint64
	DelegReturn  uint64
	GetAttr      uint64
	GetFH        uint64
	Link         uint64
	Lock         uint64
	Lockt        uint64
	Locku        uint64
	Lookup       uint64
	LookupRoot   uint64
	Nverify      uint64
	Open         uint64
	OpenAttr     uint64
	OpenConfirm  uint64
	OpenDgrd     uint64
	PutFH        uint64
	PutPubFH     uint64
	PutRootFH    uint64
	Read         uint64
	ReadDir      uint64
	ReadLink     uint64
	Remove       uint64
	Rename       uint64
	Renew        uint64
	RestoreFH    uint64
	SaveFH       uint64
	SecInfo      uint64
	SetAttr      uint64
	Verify       uint64
	Write        uint64
	RelLockOwner uint64
}

// ClientRPCStats models all stats from /proc/net/rpc/nfs.
type ClientRPCStats struct {
	Network       Network
	ClientRPC     ClientRPC
	V2Stats       V2Stats
	V3Stats       V3Stats
	ClientV4Stats ClientV4Stats
}

// ServerRPCStats models all stats from /proc/net/rpc/nfsd.
type ServerRPCStats struct {
	ReplyCache     ReplyCache
	FileHandles    FileHandles
	InputOutput    InputOutput
	Threads        Threads
	ReadAheadCache ReadAheadCache
	Network        Network
	ServerRPC      ServerRPC
	V2Stats        V2Stats
	V3Stats        V3Stats
	ServerV4Stats  ServerV4Stats
	V4Ops          V4Ops
}

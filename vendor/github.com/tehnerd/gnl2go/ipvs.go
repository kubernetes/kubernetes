package gnl2go

/*
This package implements routines to work with linux LVS, so we would be able
to work with LVS nativly, instead of using Exec(ipvsadm).
It is expecting for end user to work with this routines:

ipvs := new(IpvsClient)
err := ipvs.Init() - to init netlink socket etc

err := ipvs.Flush() - to flush lvs table

pools,err := ipvs.GetPools() - to check which services and dest has been configured

err := ipvs.AddService(vip_string,port_uint16,protocol_uint16,scheduler_string) - add new service,
	which will be described by it's address (lvs also support service by fwmark (from iptables), check bellow)
	type of service(ipv4 or ipv6) will be deduced from it's vip address

err := ipvs.DelService(vip_string,port_uint16,protocol_uint16) - to delete service

err := ipvs.AddFWMService(fwmark_uint32,sched_string,af_uint16) - add fwmark service, we must also
	provide the type of service (af; must be syscall.AF_INET for ipv4 or syscall.AF_INET6 for ipv6)

err := ipvs.DelFWMService(fwmark_uint32,af_uint16) - delete fwmark service

err := ipvs.AddDest(vip_string,port_uint16,rip_string,protocol_uint16,weight_int32) - add destination rip to vip. port on vip and rip is the same
	fwding methond - tunneling

err := ipvs.AddDestPort(vip_string,vport_uint16,rip_string,rport_uint16,protocol_uint16,weight_int32,fwd_uint32) - add destination rip to vip.
	port on vip and rip could be different. fwding method could be any supported (for example IPVS_MASQUERADING)

err := ipvs.UpdateDest(vip_string,port_uint16,rip_string,protocol_uint16,weight_int32) - change description of real server(for example
	change it's weight)

err := ipvs.UpdateDestPort(vip_string,vport_uint16,rip_string,rport_uint16, protocol_uint16,weight_int32,fwd_uint32) - same as above
	but with custom ports on real and fwd method

err := ipvs.DelDest(vip_string,port_uint16,rip_string,protocol_uint16)

err := ipvs.DelDestPort(vip_string,vport_uint16,rip_string,rport_uint16, protocol_uint16)


err := ipvs.AddFWMDest(fwmark_uint32,rip_string,vaf_uint16,port_uint16,weight_int32) - add destination to fwmark bassed service,
	vaf - fwmark's service address family.

err := ipvs.AddFWMDestFWD(fwmark_uint32,rip_string,vaf_uint16,port_uint16,weight_int32,fwd_uint32) - add destination to fwmark bassed service,
	vaf - fwmark's service address family. fwd - forwarding method (tunneling or nat/masquerading)

err := ipvs.UpdateFWMDest(fwmark_uint32,rip_string,vaf_uint16,port_uint16,weight_int32)

err := ipvs.UpdateFWMDestFWD(fwmark_uint32,rip_string,vaf_uint16,port_uint16,weight_int32,fwd_uint32)

err := DelFWMDest(fwmark_uint32,rip_string,vaf_uint16,port_uint16)

ipvs.Exit() - to close NL socket

*/

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"strconv"
	"strings"
	"syscall"
)

const (
	IPVS_MASQUERADING = 0
	IPVS_TUNNELING    = 2
)

const (
	NO_FLAGS               = 0x0    /* no flags */
	IP_VS_SVC_F_PERSISTENT = 0x0001 /* persistent port */
	IP_VS_SVC_F_HASHED     = 0x0002 /* hashed entry */
	IP_VS_SVC_F_ONEPACKET  = 0x0004 /* one-packet scheduling */
	IP_VS_SVC_F_SCHED1     = 0x0008 /* scheduler flag 1 */
	IP_VS_SVC_F_SCHED2     = 0x0010 /* scheduler flag 2 */
	IP_VS_SVC_F_SCHED3     = 0x0020 /* scheduler flag 3 */

	IP_VS_SVC_F_SCHED_SH_FALLBACK = IP_VS_SVC_F_SCHED1
	IP_VS_SVC_F_SCHED_SH_PORT     = IP_VS_SVC_F_SCHED2
)

var (
	BIN_NO_FLAGS                      = []byte{0, 0, 0, 0, 0, 0, 0, 0}        /* no flags */
	BIN_IP_VS_SVC_F_PERSISTENT        = U32ToBinFlags(IP_VS_SVC_F_PERSISTENT) /* persistent port */
	BIN_IP_VS_SVC_F_HASHED            = U32ToBinFlags(IP_VS_SVC_F_HASHED)     /* hashed entry */
	BIN_IP_VS_SVC_F_ONEPACKET         = U32ToBinFlags(IP_VS_SVC_F_ONEPACKET)  /* one-packet scheduling */
	BIN_IP_VS_SVC_F_SCHED1            = U32ToBinFlags(IP_VS_SVC_F_SCHED1)     /* scheduler flag 1 */
	BIN_IP_VS_SVC_F_SCHED2            = U32ToBinFlags(IP_VS_SVC_F_SCHED2)     /* scheduler flag 2 */
	BIN_IP_VS_SVC_F_SCHED3            = U32ToBinFlags(IP_VS_SVC_F_SCHED3)     /* scheduler flag 3 */
	BIN_IP_VS_SVC_F_SCHED_SH_FALLBACK = BIN_IP_VS_SVC_F_SCHED1
	BIN_IP_VS_SVC_F_SCHED_SH_PORT     = BIN_IP_VS_SVC_F_SCHED2
)

var (
	IpvsStatsAttrList = CreateAttrListDefinition("IpvsStatsAttrList",
		[]AttrTuple{
			AttrTuple{Name: "CONNS", Type: "U32Type"},
			AttrTuple{Name: "INPKTS", Type: "U32Type"},
			AttrTuple{Name: "OUTPKTS", Type: "U32Type"},
			AttrTuple{Name: "INBYTES", Type: "U64Type"},
			AttrTuple{Name: "OUTBYTES", Type: "U64Type"},
			AttrTuple{Name: "CPS", Type: "U32Type"},
			AttrTuple{Name: "INPPS", Type: "U32Type"},
			AttrTuple{Name: "OUTPPS", Type: "U32Type"},
			AttrTuple{Name: "INBPS", Type: "U32Type"},
			AttrTuple{Name: "OUTBPS", Type: "U32Type"},
		})

	IpvsStats64AttrList = CreateAttrListDefinition("IpvsStats64AttrList",
		[]AttrTuple{
			AttrTuple{Name: "CONNS", Type: "U64Type"},
			AttrTuple{Name: "INPKTS", Type: "U64Type"},
			AttrTuple{Name: "OUTPKTS", Type: "U64Type"},
			AttrTuple{Name: "INBYTES", Type: "U64Type"},
			AttrTuple{Name: "OUTBYTES", Type: "U64Type"},
			AttrTuple{Name: "CPS", Type: "U64Type"},
			AttrTuple{Name: "INPPS", Type: "U64Type"},
			AttrTuple{Name: "OUTPPS", Type: "U64Type"},
			AttrTuple{Name: "INBPS", Type: "U64Type"},
			AttrTuple{Name: "OUTBPS", Type: "U64Type"},
		})

	IpvsServiceAttrList = CreateAttrListDefinition("IpvsServiceAttrList",
		[]AttrTuple{
			AttrTuple{Name: "AF", Type: "U16Type"},
			AttrTuple{Name: "PROTOCOL", Type: "U16Type"},
			AttrTuple{Name: "ADDR", Type: "BinaryType"},
			AttrTuple{Name: "PORT", Type: "Net16Type"},
			AttrTuple{Name: "FWMARK", Type: "U32Type"},
			AttrTuple{Name: "SCHED_NAME", Type: "NulStringType"},
			AttrTuple{Name: "FLAGS", Type: "BinaryType"},
			AttrTuple{Name: "TIMEOUT", Type: "U32Type"},
			AttrTuple{Name: "NETMASK", Type: "U32Type"},
			AttrTuple{Name: "STATS", Type: "IpvsStatsAttrList"},
			AttrTuple{Name: "PE_NAME", Type: "NulStringType"},
			AttrTuple{Name: "STATS64", Type: "IpvsStats64AttrList"},
		})

	IpvsDestAttrList = CreateAttrListDefinition("IpvsDestAttrList",
		[]AttrTuple{
			AttrTuple{Name: "ADDR", Type: "BinaryType"},
			AttrTuple{Name: "PORT", Type: "Net16Type"},
			AttrTuple{Name: "FWD_METHOD", Type: "U32Type"},
			AttrTuple{Name: "WEIGHT", Type: "I32Type"},
			AttrTuple{Name: "U_THRESH", Type: "U32Type"},
			AttrTuple{Name: "L_THRESH", Type: "U32Type"},
			AttrTuple{Name: "ACTIVE_CONNS", Type: "U32Type"},
			AttrTuple{Name: "INACT_CONNS", Type: "U32Type"},
			AttrTuple{Name: "PERSIST_CONNS", Type: "U32Type"},
			AttrTuple{Name: "STATS", Type: "IpvsStatsAttrList"},
			AttrTuple{Name: "ADDR_FAMILY", Type: "U16Type"},
			AttrTuple{Name: "STATS64", Type: "IpvsStats64AttrList"},
		})

	IpvsDaemonAttrList = CreateAttrListDefinition("IpvsDaemonAttrList",
		[]AttrTuple{
			AttrTuple{Name: "STATE", Type: "U32Type"},
			AttrTuple{Name: "MCAST_IFN", Type: "NulStringType"},
			AttrTuple{Name: "SYNC_ID", Type: "U32Type"},
		})

	IpvsInfoAttrList = CreateAttrListDefinition("IpvsInfoAttrList",
		[]AttrTuple{
			AttrTuple{Name: "VERSION", Type: "U32Type"},
			AttrTuple{Name: "CONN_TAB_SIZE", Type: "U32Type"},
		})

	IpvsCmdAttrList = CreateAttrListDefinition("IpvsCmdAttrList",
		[]AttrTuple{
			AttrTuple{Name: "SERVICE", Type: "IpvsServiceAttrList"},
			AttrTuple{Name: "DEST", Type: "IpvsDestAttrList"},
			AttrTuple{Name: "DAEMON", Type: "IpvsDaemonAttrList"},
			AttrTuple{Name: "TIMEOUT_TCP", Type: "U32Type"},
			AttrTuple{Name: "TIMEOUT_TCP_FIN", Type: "U32Type"},
			AttrTuple{Name: "TIMEOUT_UDP", Type: "U32Type"},
		})

	IpvsMessageInitList = []AttrListTuple{
		AttrListTuple{Name: "NEW_SERVICE", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "SET_SERVICE", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "DEL_SERVICE", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "GET_SERVICE", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "NEW_DEST", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "SET_DEST", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "DEL_DEST", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "GET_DEST", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "NEW_DAEMON", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "DEL_DAEMON", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "GET_DAEMON", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "SET_CONFIG", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "GET_CONFIG", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "SET_INFO", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "GET_INFO", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "ZERO", AttrList: CreateAttrListType(IpvsCmdAttrList)},
		AttrListTuple{Name: "FLUSH", AttrList: CreateAttrListType(IpvsCmdAttrList)},
	}
)

func U32ToBinFlags(flags uint32) []byte {
	buf := new(bytes.Buffer)
	err := binary.Write(buf, binary.LittleEndian, flags)
	if err != nil {
		/* we dont wanna trow so will return all nulls */
		return []byte{0, 0, 0, 0, 0, 0, 0, 0}
	}
	encFlags := buf.Bytes()
	/* this is helper function and was tested w/ current flags (as for 4.5)
	   so mask will cover only first byte; as for now, there is no flags
	   in subsequent */
	encFlags = append(encFlags, []byte{255, 0, 0, 0}...)
	return encFlags
}

func validateIp(ip string) bool {
	for _, c := range ip {
		if c == ':' {
			_, err := IPv6StringToAddr(ip)
			if err != nil {
				return false
			}
			return true
		}
	}
	_, err := IPv4ToUint32(ip)
	if err != nil {
		return false
	}
	return true
}

func toAFUnion(ip string) (uint16, []byte, error) {
	buf := new(bytes.Buffer)
	for _, c := range ip {
		if c == ':' {
			addr, _ := IPv6StringToAddr(ip)
			err := binary.Write(buf, binary.BigEndian, addr)
			if err != nil {
				return 0, nil, err
			}
			encAddr := buf.Bytes()
			if len(encAddr) != 16 {
				return 0, nil, fmt.Errorf("length not equal to 16\n")
			}
			return syscall.AF_INET6, encAddr, nil
		}
	}
	addr, err := IPv4ToUint32(ip)
	if err != nil {
		return 0, nil, err
	}
	err = binary.Write(buf, binary.BigEndian, addr)
	if err != nil {
		return 0, nil, err
	}
	encAddr := buf.Bytes()
	for len(encAddr) != 16 {
		encAddr = append(encAddr, byte(0))
	}
	return syscall.AF_INET, encAddr, nil
}

func fromAFUnion(af uint16, addr []byte) (string, error) {
	if af == syscall.AF_INET6 {
		var v6addr IPv6Addr
		err := binary.Read(bytes.NewReader(addr), binary.BigEndian, &v6addr)
		if err != nil {
			return "", fmt.Errorf("cant decode ipv6 addr from net repr:%v\n", err)
		}
		addrStr := IPv6AddrToString(v6addr)
		return addrStr, nil
	}
	var v4addr uint32
	//we leftpadded addr to len 16 above,so our v4 addr in addr[:4]
	err := binary.Read(bytes.NewReader(addr[:4]), binary.BigEndian, &v4addr)
	if err != nil {
		return "", fmt.Errorf("cant decode ipv4 addr from net repr:%v\n", err)
	}
	addrStr := Uint32IPv4ToString(v4addr)
	return addrStr, nil
}

func ToProtoNum(proto NulStringType) U16Type {
	p := string(proto)
	switch strings.ToLower(p) {
	case "tcp":
		return U16Type(syscall.IPPROTO_TCP)
	case "udp":
		return U16Type(syscall.IPPROTO_UDP)
	}
	return U16Type(0)
}

func FromProtoNum(pnum U16Type) NulStringType {
	switch uint16(pnum) {
	case syscall.IPPROTO_TCP:
		return NulStringType("TCP")
	case syscall.IPPROTO_UDP:
		return NulStringType("UDP")
	}
	return NulStringType("UNKNOWN")
}

type Dest struct {
	IP     string
	Weight int32
	Port   uint16
	AF     uint16
}

func (d *Dest) IsEqual(od *Dest) bool {
	return d.IP == od.IP && d.Weight == od.Weight && d.Port == od.Port
}

func (d *Dest) InitFromAttrList(list map[string]SerDes) error {
	//lots of casts from interface w/o checks; so we are going to panic if something goes wrong
	af, ok := list["ADDR_FAMILY"].(*U16Type)
	if !ok {
		//OLD kernel (3.18-), which doesnt support addr_family in dest definition
		dAF := U16Type(d.AF)
		af = &dAF
	} else {
		d.AF = uint16(*af)
	}
	addr, ok := list["ADDR"].(*BinaryType)
	if !ok {
		return fmt.Errorf("no dst ADDR in attr list: %#v\n", list)
	}
	ip, err := fromAFUnion(uint16(*af), []byte(*addr))
	if err != nil {
		return err
	}
	d.IP = ip
	w, ok := list["WEIGHT"].(*I32Type)
	if !ok {
		return fmt.Errorf("no dst WEIGHT in attr list: %#v\n", list)
	}
	d.Weight = int32(*w)
	p, ok := list["PORT"].(*Net16Type)
	if !ok {
		return fmt.Errorf("no dst PORT in attr list: %#v\n", list)
	}
	d.Port = uint16(*p)
	return nil
}

type Service struct {
	Proto  uint16
	VIP    string
	Port   uint16
	Sched  string
	FWMark uint32
	AF     uint16
}

func (s *Service) IsEqual(os Service) bool {
	return s.Proto == os.Proto && s.VIP == os.VIP &&
		s.Port == os.Port && s.Sched == os.Sched && s.FWMark == os.FWMark
}

func (s *Service) InitFromAttrList(list map[string]SerDes) error {
	if _, exists := list["ADDR"]; exists {
		af := list["AF"].(*U16Type)
		s.AF = uint16(*af)
		addr := list["ADDR"].(*BinaryType)
		vip, err := fromAFUnion(uint16(*af), []byte(*addr))
		if err != nil {
			return err
		}
		s.VIP = vip
		proto := list["PROTOCOL"].(*U16Type)
		s.Proto = uint16(*proto)
		p := list["PORT"].(*Net16Type)
		s.Port = uint16(*p)
	} else {
		fw := list["FWMARK"].(*U32Type)
		s.FWMark = uint32(*fw)
		af := list["AF"].(*U16Type)
		s.AF = uint16(*af)
	}
	sched := list["SCHED_NAME"].(*NulStringType)
	s.Sched = string(*sched)
	return nil
}

func (s *Service) CreateAttrList() (map[string]SerDes, error) {
	attrList := make(map[string]SerDes)
	if s.VIP != "" {
		//this is not FW Mark based service
		af, addr, err := toAFUnion(s.VIP)
		if err != nil {
			return nil, err
		}
		//1<<32-1
		netmask := uint32(4294967295)
		if af == syscall.AF_INET6 {
			netmask = 128
		}
		AF := U16Type(af)
		Port := Net16Type(s.Port)
		Netmask := U32Type(netmask)
		Addr := BinaryType(addr)
		Proto := U16Type(s.Proto)
		attrList["AF"] = &AF
		attrList["PORT"] = &Port
		attrList["PROTOCOL"] = &Proto
		attrList["ADDR"] = &Addr
		attrList["NETMASK"] = &Netmask
	} else {
		//FW Mark
		FWMark := U32Type(s.FWMark)
		AF := U16Type(s.AF)
		attrList["FWMARK"] = &FWMark
		attrList["AF"] = &AF
	}
	Sched := NulStringType(s.Sched)
	attrList["SCHED_NAME"] = &Sched
	return attrList, nil
}

func (s *Service) ToString() string {
	if s.VIP != "" {
		port := strconv.FormatUint(uint64(s.Port), 10)
		proto := ""
		switch s.Proto {
		case syscall.IPPROTO_TCP:
			proto = "tcp"
		case syscall.IPPROTO_UDP:
			proto = "udp"
		default:
			proto = "unknown"
		}
		return strings.Join([]string{s.VIP, proto, port}, ":")
	} else {
		//fwmark based service
		fwmark := strconv.FormatUint(uint64(s.FWMark), 10)
		proto := ""
		switch s.AF {
		case syscall.AF_INET:
			proto = "ipv4"
		case syscall.AF_INET6:
			proto = "ipv6"
		default:
			proto = "unknown"
		}
		return strings.Join([]string{"fwmark", proto, fwmark}, ":")
	}
}

type Pool struct {
	Service Service
	Dests   []Dest
}

func (p *Pool) InitFromAttrList(list map[string]SerDes) {
	//TODO(tehnerd):...
}

type StatsIntf interface {
	GetStats() map[string]uint64
}

type Stats struct {
	Conns    uint32
	Inpkts   uint32
	Outpkts  uint32
	Inbytes  uint64
	Outbytes uint64
	Cps      uint32
	Inpps    uint32
	Outpps   uint32
	Inbps    uint32
	Outbps   uint32
}

func (stats *Stats) InitFromAttrList(list map[string]SerDes) {
	//not tested
	conns := list["CONNS"].(*U32Type)
	inpkts := list["INPKTS"].(*U32Type)
	outpkts := list["OUTPKTS"].(*U32Type)
	inbytes := list["INBYTES"].(*U64Type)
	outbytes := list["OUTBYTES"].(*U64Type)
	cps := list["CPS"].(*U32Type)
	inpps := list["INPPS"].(*U32Type)
	outpps := list["OUTPPS"].(*U32Type)
	inbps := list["INBPS"].(*U32Type)
	outbps := list["OUTBPS"].(*U32Type)
	stats.Conns = uint32(*conns)
	stats.Inpkts = uint32(*inpkts)
	stats.Outpkts = uint32(*outpkts)
	stats.Inbytes = uint64(*inbytes)
	stats.Outbytes = uint64(*outbytes)
	stats.Cps = uint32(*cps)
	stats.Inpps = uint32(*inpps)
	stats.Outpps = uint32(*outpps)
	stats.Inbps = uint32(*inbps)
	stats.Outbps = uint32(*outbps)
}

func (stats Stats) GetStats() map[string]uint64 {
	statsMap := make(map[string]uint64)
	statsMap["CONNS"] = uint64(stats.Conns)
	statsMap["INPKTS"] = uint64(stats.Inpkts)
	statsMap["OUTPKTS"] = uint64(stats.Outpkts)
	statsMap["INBYTES"] = uint64(stats.Inbytes)
	statsMap["OUTBYTES"] = uint64(stats.Outbytes)
	statsMap["CPS"] = uint64(stats.Cps)
	statsMap["INPPS"] = uint64(stats.Inpps)
	statsMap["OUTPPS"] = uint64(stats.Outpps)
	statsMap["INBPS"] = uint64(stats.Inbps)
	statsMap["OUTBPS"] = uint64(stats.Outbps)
	return statsMap
}

type Stats64 struct {
	Conns    uint64
	Inpkts   uint64
	Outpkts  uint64
	Inbytes  uint64
	Outbytes uint64
	Cps      uint64
	Inpps    uint64
	Outpps   uint64
	Inbps    uint64
	Outbps   uint64
}

func (stats *Stats64) InitFromAttrList(list map[string]SerDes) {
	//not tested
	conns := list["CONNS"].(*U64Type)
	inpkts := list["INPKTS"].(*U64Type)
	outpkts := list["OUTPKTS"].(*U64Type)
	inbytes := list["INBYTES"].(*U64Type)
	outbytes := list["OUTBYTES"].(*U64Type)
	cps := list["CPS"].(*U64Type)
	inpps := list["INPPS"].(*U64Type)
	outpps := list["OUTPPS"].(*U64Type)
	inbps := list["INBPS"].(*U64Type)
	outbps := list["OUTBPS"].(*U64Type)
	stats.Conns = uint64(*conns)
	stats.Inpkts = uint64(*inpkts)
	stats.Outpkts = uint64(*outpkts)
	stats.Inbytes = uint64(*inbytes)
	stats.Outbytes = uint64(*outbytes)
	stats.Cps = uint64(*cps)
	stats.Inpps = uint64(*inpps)
	stats.Outpps = uint64(*outpps)
	stats.Inbps = uint64(*inbps)
	stats.Outbps = uint64(*outbps)
}

func (stats Stats64) GetStats() map[string]uint64 {
	statsMap := make(map[string]uint64)
	statsMap["CONNS"] = stats.Conns
	statsMap["INPKTS"] = stats.Inpkts
	statsMap["OUTPKTS"] = stats.Outpkts
	statsMap["INBYTES"] = stats.Inbytes
	statsMap["OUTBYTES"] = stats.Outbytes
	statsMap["CPS"] = stats.Cps
	statsMap["INPPS"] = stats.Inpps
	statsMap["OUTPPS"] = stats.Outpps
	statsMap["INBPS"] = stats.Inbps
	statsMap["OUTBPS"] = stats.Outbps
	return statsMap
}

type IpvsClient struct {
	Sock NLSocket
	mt   *MessageType
}

func (ipvs *IpvsClient) Init() error {
	LookupTypeOnStartup(IpvsMessageInitList, "IPVS")
	err := ipvs.Sock.Init()
	if err != nil {
		return err
	}
	ipvs.mt = Family2MT[MT2Family["IPVS"]]
	return nil
}

func (ipvs *IpvsClient) Flush() error {
	msg, err := ipvs.mt.InitGNLMessageStr("FLUSH", ACK_REQUEST)
	if err != nil {
		return err
	}
	err = ipvs.Sock.Execute(msg)
	if err != nil {
		return err
	}
	return nil
}

func (ipvs *IpvsClient) GetPoolForService(svc Service) (Pool, error) {
	attrList, err := svc.CreateAttrList()
	if err != nil {
		return Pool{}, err
	}
	pool, err := ipvs.getPoolForAttrList(attrList)
	return pool, err
}

func (ipvs *IpvsClient) getPoolForAttrList(
	list map[string]SerDes) (Pool, error) {
	var pool Pool
	pool.Service.InitFromAttrList(list)
	destReq, err := ipvs.mt.InitGNLMessageStr("GET_DEST", MATCH_ROOT_REQUEST)
	if err != nil {
		return pool, err
	}
	svcAttrListDef, _ := ATLName2ATL["IpvsServiceAttrList"]
	svcAttrListType := CreateAttrListType(svcAttrListDef)
	svcAttrListType.Set(list)
	destReq.AttrMap["SERVICE"] = &svcAttrListType
	destResps, err := ipvs.Sock.Query(destReq)
	if err != nil {
		return pool, err
	}
	for _, destResp := range destResps {
		var d Dest
		dstAttrList := destResp.GetAttrList("DEST")
		d.AF = pool.Service.AF
		if dstAttrList != nil {
			d.InitFromAttrList(dstAttrList.(*AttrListType).Amap)
			pool.Dests = append(pool.Dests, d)
		}
	}
	return pool, nil
}

func (ipvs *IpvsClient) GetPools() ([]Pool, error) {
	var pools []Pool
	msg, err := ipvs.mt.InitGNLMessageStr("GET_SERVICE", MATCH_ROOT_REQUEST)
	if err != nil {
		return nil, err
	}
	resps, err := ipvs.Sock.Query(msg)
	if err != nil {
		return nil, err
	}
	for _, resp := range resps {
		svcAttrList := resp.GetAttrList("SERVICE")
		pool, err := ipvs.getPoolForAttrList(svcAttrList.(*AttrListType).Amap)
		if err != nil {
			return nil, err
		}
		pools = append(pools, pool)
	}
	return pools, nil
}

func GetStatsFromAttrList(attrList *AttrListType) StatsIntf {
	if val, exists := attrList.Amap["STATS64"]; exists {
		var sstats64 Stats64
		sstats64.InitFromAttrList(val.(*AttrListType).Amap)
		return sstats64
	} else {
		var sstats Stats
		statsAttrList := attrList.Amap["STATS"]
		sstats.InitFromAttrList(statsAttrList.(*AttrListType).Amap)
		return sstats
	}
	//we should never reach this
	panic("check GetStatsFromAttrList routine")
	return nil
}

func (ipvs *IpvsClient) GetAllStatsBrief() (map[string]StatsIntf, error) {
	statsMap := make(map[string]StatsIntf)
	msg, err := ipvs.mt.InitGNLMessageStr("GET_SERVICE", MATCH_ROOT_REQUEST)
	if err != nil {
		return nil, err
	}
	resps, err := ipvs.Sock.Query(msg)
	if err != nil {
		return nil, err
	}
	for _, resp := range resps {
		var svc Service
		svcAttrList := resp.GetAttrList("SERVICE").(*AttrListType)
		svc.InitFromAttrList(svcAttrList.Amap)
		sstat := GetStatsFromAttrList(svcAttrList)
		statsMap[svc.ToString()] = sstat
	}
	return statsMap, nil
}

func (ipvs *IpvsClient) modifyService(method string, vip string,
	port uint16, protocol uint16, amap map[string]SerDes) error {
	af, addr, err := toAFUnion(vip)
	if err != nil {
		return err
	}
	//1<<32-1
	netmask := uint32(4294967295)
	if af == syscall.AF_INET6 {
		netmask = 128
	}
	msg, err := ipvs.mt.InitGNLMessageStr(method, ACK_REQUEST)
	if err != nil {
		return err
	}
	AF := U16Type(af)
	Port := Net16Type(port)
	Netmask := U32Type(netmask)
	Addr := BinaryType(addr)
	Proto := U16Type(protocol)
	atl, _ := ATLName2ATL["IpvsServiceAttrList"]
	sattr := CreateAttrListType(atl)
	sattr.Amap["AF"] = &AF
	sattr.Amap["PORT"] = &Port
	sattr.Amap["PROTOCOL"] = &Proto
	sattr.Amap["ADDR"] = &Addr
	sattr.Amap["NETMASK"] = &Netmask
	for k, v := range amap {
		sattr.Amap[k] = v
	}
	msg.AttrMap["SERVICE"] = &sattr
	err = ipvs.Sock.Execute(msg)
	if err != nil {
		return err
	}
	return nil
}

func (ipvs *IpvsClient) AddService(vip string,
	port uint16, protocol uint16, sched string) error {
	return ipvs.AddServiceWithFlags(vip, port, protocol,
		sched, BIN_NO_FLAGS)
}

func (ipvs *IpvsClient) AddServiceWithFlags(vip string,
	port uint16, protocol uint16, sched string, flags []byte) error {
	paramsMap := make(map[string]SerDes)
	Sched := NulStringType(sched)
	Timeout := U32Type(0)
	Flags := BinaryType(flags)
	paramsMap["FLAGS"] = &Flags
	paramsMap["SCHED_NAME"] = &Sched
	paramsMap["TIMEOUT"] = &Timeout
	err := ipvs.modifyService("NEW_SERVICE", vip, port,
		protocol, paramsMap)
	if err != nil {
		return err
	}
	return nil
}

func (ipvs *IpvsClient) DelService(vip string,
	port uint16, protocol uint16) error {
	err := ipvs.modifyService("DEL_SERVICE", vip, port,
		protocol, nil)
	if err != nil {
		return err
	}
	return nil
}

func (ipvs *IpvsClient) modifyFWMService(method string, fwmark uint32,
	af uint16, amap map[string]SerDes) error {
	AF := U16Type(af)
	FWMark := U32Type(fwmark)
	netmask := uint32(4294967295)
	if af == syscall.AF_INET6 {
		netmask = 128
	}
	msg, err := ipvs.mt.InitGNLMessageStr(method, ACK_REQUEST)
	if err != nil {
		return err
	}
	Netmask := U32Type(netmask)
	atl, _ := ATLName2ATL["IpvsServiceAttrList"]
	sattr := CreateAttrListType(atl)
	sattr.Amap["FWMARK"] = &FWMark
	sattr.Amap["AF"] = &AF
	sattr.Amap["NETMASK"] = &Netmask
	for k, v := range amap {
		sattr.Amap[k] = v
	}
	msg.AttrMap["SERVICE"] = &sattr
	err = ipvs.Sock.Execute(msg)
	if err != nil {
		return err
	}
	return nil
}

func (ipvs *IpvsClient) AddFWMService(fwmark uint32,
	sched string, af uint16) error {
	return ipvs.AddFWMServiceWithFlags(fwmark, sched, af, BIN_NO_FLAGS)
}

func (ipvs *IpvsClient) AddFWMServiceWithFlags(fwmark uint32,
	sched string, af uint16, flags []byte) error {
	paramsMap := make(map[string]SerDes)
	Sched := NulStringType(sched)
	Timeout := U32Type(0)
	Flags := BinaryType(flags)
	paramsMap["FLAGS"] = &Flags
	paramsMap["SCHED_NAME"] = &Sched
	paramsMap["TIMEOUT"] = &Timeout
	err := ipvs.modifyFWMService("NEW_SERVICE", fwmark,
		af, paramsMap)
	if err != nil {
		return err
	}
	return nil
}

func (ipvs *IpvsClient) DelFWMService(fwmark uint32, af uint16) error {
	err := ipvs.modifyFWMService("DEL_SERVICE", fwmark, af, nil)
	if err != nil {
		return err
	}
	return nil
}

func (ipvs *IpvsClient) modifyDest(method string, vip string, vport uint16,
	rip string, rport uint16, protocol uint16, amap map[string]SerDes) error {
	//starts with r - for real's related, v - for vip's
	vaf, vaddr, err := toAFUnion(vip)
	if err != nil {
		return err
	}
	raf, raddr, err := toAFUnion(rip)
	if err != nil {
		return err
	}
	msg, err := ipvs.mt.InitGNLMessageStr(method, ACK_REQUEST)
	if err != nil {
		return err
	}
	vAF := U16Type(vaf)
	vAddr := BinaryType(vaddr)
	rAF := U16Type(raf)
	rAddr := BinaryType(raddr)

	VPort := Net16Type(vport)
	RPort := Net16Type(rport)
	Proto := U16Type(protocol)

	vatl, _ := ATLName2ATL["IpvsServiceAttrList"]
	ratl, _ := ATLName2ATL["IpvsDestAttrList"]
	sattr := CreateAttrListType(vatl)
	rattr := CreateAttrListType(ratl)

	sattr.Amap["AF"] = &vAF
	sattr.Amap["PORT"] = &VPort
	sattr.Amap["PROTOCOL"] = &Proto
	sattr.Amap["ADDR"] = &vAddr

	/*
		XXX(tehnerd): real's port right now is equal to vip's but again it's trivial to fix
		for example in param map you could override amap["PORT"]
	*/
	rattr.Amap["ADDR_FAMILY"] = &rAF
	rattr.Amap["PORT"] = &RPort
	rattr.Amap["ADDR"] = &rAddr

	for k, v := range amap {
		rattr.Amap[k] = v
	}
	msg.AttrMap["SERVICE"] = &sattr
	msg.AttrMap["DEST"] = &rattr
	err = ipvs.Sock.Execute(msg)
	if err != nil {
		return err
	}
	return nil

}

func (ipvs *IpvsClient) AddDest(vip string, port uint16, rip string,
	protocol uint16, weight int32) error {
	return ipvs.AddDestPort(vip, port, rip, port, protocol, weight, IPVS_TUNNELING)
}

func (ipvs *IpvsClient) AddDestPort(vip string, vport uint16, rip string,
	rport uint16, protocol uint16, weight int32, fwd uint32) error {
	paramsMap := make(map[string]SerDes)
	Weight := I32Type(weight)
	//XXX(tehnerd): hardcode, but easy to fix; 2 - tunneling
	FWDMethod := U32Type(fwd)
	LThresh := U32Type(0)
	UThresh := U32Type(0)
	paramsMap["WEIGHT"] = &Weight
	paramsMap["FWD_METHOD"] = &FWDMethod
	paramsMap["L_THRESH"] = &LThresh
	paramsMap["U_THRESH"] = &UThresh
	err := ipvs.modifyDest("NEW_DEST", vip, vport, rip, rport, protocol, paramsMap)
	if err != nil {
		return err
	}
	return err
}

func (ipvs *IpvsClient) UpdateDest(vip string, port uint16, rip string,
	protocol uint16, weight int32) error {
	return ipvs.UpdateDestPort(vip, port, rip, port, protocol, weight, IPVS_TUNNELING)
}

func (ipvs *IpvsClient) UpdateDestPort(vip string, vport uint16, rip string,
	rport uint16, protocol uint16, weight int32, fwd uint32) error {
	paramsMap := make(map[string]SerDes)
	Weight := I32Type(weight)
	//XXX(tehnerd): hardcode, but easy to fix; 2 - tunneling
	FWDMethod := U32Type(fwd)
	LThresh := U32Type(0)
	UThresh := U32Type(0)
	paramsMap["WEIGHT"] = &Weight
	paramsMap["FWD_METHOD"] = &FWDMethod
	paramsMap["L_THRESH"] = &LThresh
	paramsMap["U_THRESH"] = &UThresh
	err := ipvs.modifyDest("SET_DEST", vip, vport, rip, rport, protocol, paramsMap)
	if err != nil {
		return err
	}
	return nil
}

func (ipvs *IpvsClient) DelDest(vip string, port uint16, rip string,
	protocol uint16) error {
	return ipvs.DelDestPort(vip, port, rip, port, protocol)
}

func (ipvs *IpvsClient) DelDestPort(vip string, vport uint16, rip string,
	rport uint16, protocol uint16) error {
	err := ipvs.modifyDest("DEL_DEST", vip, vport, rip, rport, protocol, nil)
	if err != nil {
		return err
	}
	return nil
}

func (ipvs *IpvsClient) modifyFWMDest(method string, fwmark uint32,
	rip string, vaf uint16, port uint16, amap map[string]SerDes) error {
	//starts with r - for real's related, v - for vip's
	raf, raddr, err := toAFUnion(rip)
	if err != nil {
		return err
	}
	msg, err := ipvs.mt.InitGNLMessageStr(method, ACK_REQUEST)
	if err != nil {
		return err
	}
	vAF := U16Type(vaf)

	rAF := U16Type(raf)
	rAddr := BinaryType(raddr)
	Port := Net16Type(port)

	FWMark := U32Type(fwmark)

	vatl, _ := ATLName2ATL["IpvsServiceAttrList"]
	ratl, _ := ATLName2ATL["IpvsDestAttrList"]

	sattr := CreateAttrListType(vatl)
	rattr := CreateAttrListType(ratl)

	sattr.Amap["FWMARK"] = &FWMark
	sattr.Amap["AF"] = &vAF

	rattr.Amap["ADDR_FAMILY"] = &rAF
	rattr.Amap["ADDR"] = &rAddr
	rattr.Amap["PORT"] = &Port

	for k, v := range amap {
		rattr.Amap[k] = v
	}
	msg.AttrMap["SERVICE"] = &sattr
	msg.AttrMap["DEST"] = &rattr
	err = ipvs.Sock.Execute(msg)
	if err != nil {
		return err
	}
	return nil
}

/*
func (ipvs *IpvsClient) modifyFWMDest(method string, fwmark uint32,
	rip string, vaf uint16, port uint16, amap map[string]SerDes) {

*/

func (ipvs *IpvsClient) AddFWMDest(fwmark uint32, rip string, vaf uint16,
	port uint16, weight int32) error {
	return ipvs.AddFWMDestFWD(fwmark, rip, vaf, port, weight, IPVS_TUNNELING)
}

func (ipvs *IpvsClient) AddFWMDestFWD(fwmark uint32, rip string, vaf uint16,
	port uint16, weight int32, fwd uint32) error {
	paramsMap := make(map[string]SerDes)
	Weight := I32Type(weight)
	//XXX(tehnerd): hardcode, but easy to fix; 2 - tunneling
	FWDMethod := U32Type(fwd)
	LThresh := U32Type(0)
	UThresh := U32Type(0)
	paramsMap["WEIGHT"] = &Weight
	paramsMap["FWD_METHOD"] = &FWDMethod
	paramsMap["L_THRESH"] = &LThresh
	paramsMap["U_THRESH"] = &UThresh
	err := ipvs.modifyFWMDest("NEW_DEST", fwmark, rip, vaf, port, paramsMap)
	if err != nil {
		return err
	}
	return nil
}
func (ipvs *IpvsClient) UpdateFWMDest(fwmark uint32, rip string, vaf uint16,
	port uint16, weight int32) error {
	return ipvs.UpdateFWMDestFWD(fwmark, rip, vaf, port, weight, IPVS_TUNNELING)
}

func (ipvs *IpvsClient) UpdateFWMDestFWD(fwmark uint32, rip string, vaf uint16,
	port uint16, weight int32, fwd uint32) error {
	paramsMap := make(map[string]SerDes)
	Weight := I32Type(weight)
	//XXX(tehnerd): hardcode, but easy to fix; 2 - tunneling
	FWDMethod := U32Type(fwd)
	LThresh := U32Type(0)
	UThresh := U32Type(0)
	paramsMap["WEIGHT"] = &Weight
	paramsMap["FWD_METHOD"] = &FWDMethod
	paramsMap["L_THRESH"] = &LThresh
	paramsMap["U_THRESH"] = &UThresh
	err := ipvs.modifyFWMDest("SET_DEST", fwmark, rip, vaf, port, paramsMap)
	if err != nil {
		return err
	}
	return nil
}

func (ipvs *IpvsClient) DelFWMDest(fwmark uint32, rip string, vaf uint16,
	port uint16) error {
	err := ipvs.modifyFWMDest("DEL_DEST", fwmark, rip, vaf, port, nil)
	if err != nil {
		return err
	}
	return nil
}

func (ipvs *IpvsClient) Exit() {
	ipvs.Sock.Close()
}

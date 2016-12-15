package gnl2go

/*
This package implements routines to work with Linux Generic Netlink Sockets
There is a lot of generic netlink's specific parsing and hacks (coz of nature how nl is implemented;
more info could be found @ rfc 3549 and man 7 netlink)
For most of the ppl working with this lib will looks like(for example you can look into ipvs.go,
which implements Linux'es LVS related wraper and routines):

Define Netlink's Attribute's Lists and Message Types for particular family

Create some kind of wraper around NLSocket,
which, during init proccess, will add it's family name to LookupOnStartup Dict.

Then the user of the lib would  create generic netlink message
(InitGNLMessageStr, inited with msg name from MessageType dict)and add coresponding
Attributes dict (msg.AttrMap)to that msg

After that the user would either NLSocket.Execute(msg) or NLSocket.Query(msg).
former will be used in case when we dont need any feedback, the latter would be used if we are quering for some kind of
info from the kernel (the return type would be GNLMessage, and you can loop around AttrMap of that message)
*/

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"sync"
	"syscall"
)

const (
	REQUEST   = 1
	MULTI     = 2
	ACK       = 4
	ECHO      = 8
	DUMP_INTR = 16

	ROOT   = 0x100
	MATCH  = 0x200
	ATOMIC = 0x400
	DUMP   = (ROOT | MATCH)

	REPLACE = 0x100
	EXCL    = 0x200
	CREATE  = 0x400
	APPEND  = 0x800

	NETLINK_GENERIC = 16

	ACK_REQUEST        = (REQUEST | ACK)
	MATCH_ROOT_REQUEST = (MATCH | ROOT | REQUEST)
)

/*
from gnlpy:
	In order to discover family IDs, we'll need to exchange some Ctrl
messages with the kernel.  We declare these message types and attribute
list types below.
*/

var (
	CtrlOpsAttrList = CreateAttrListDefinition("CtrlOpsAttrList",
		[]AttrTuple{
			AttrTuple{Name: "ID", Type: "U32Type"},
			AttrTuple{Name: "FLAGS", Type: "U32Type"},
		})

	CtrlMcastGroupAttrList = CreateAttrListDefinition("CtrlMcastGroupAttrList",
		[]AttrTuple{
			AttrTuple{Name: "NAME", Type: "NulStringType"},
			AttrTuple{Name: "ID", Type: "U32Type"},
		})

	CtrlAttrList = CreateAttrListDefinition("CtrlAttrList",
		[]AttrTuple{
			AttrTuple{Name: "FAMILY_ID", Type: "U16Type"},
			AttrTuple{Name: "FAMILY_NAME", Type: "NulStringType"},
			AttrTuple{Name: "VERSION", Type: "U32Type"},
			AttrTuple{Name: "HDRSIZE", Type: "U32Type"},
			AttrTuple{Name: "MAXATTR", Type: "U32Type"},
			AttrTuple{Name: "OPS", Type: "IgnoreType"},
			AttrTuple{Name: "MCAST_GROUPS", Type: "CtrlMcastGroupAttrList"},
		})

	NoneAttrList = []AttrTuple{}

	CtrlMessageInitList = []AttrListTuple{
		AttrListTuple{Name: "NEWFAMILY", AttrList: CreateAttrListType(CtrlAttrList)},
		AttrListTuple{Name: "DELFAMILY", AttrList: CreateAttrListType(NoneAttrList)},
		AttrListTuple{Name: "GETFAMILY", AttrList: CreateAttrListType(CtrlAttrList)},
		AttrListTuple{Name: "NEWOPS", AttrList: CreateAttrListType(NoneAttrList)},
		AttrListTuple{Name: "DELOPS", AttrList: CreateAttrListType(NoneAttrList)},
		AttrListTuple{Name: "GETOPS", AttrList: CreateAttrListType(NoneAttrList)},
		AttrListTuple{Name: "NEWMCAST_GRP", AttrList: CreateAttrListType(NoneAttrList)},
		AttrListTuple{Name: "DELMCAST_GRP", AttrList: CreateAttrListType(NoneAttrList)},
		AttrListTuple{Name: "GETMCAST_GRP", AttrList: CreateAttrListType(NoneAttrList)},
	}
	ErrorMessageInitList = []AttrListTuple{}
	DoneMessageInitList  = []AttrListTuple{}
	CtrlMessage          = CreateMsgType(CtrlMessageInitList, 16)
	ErrorMessage         = CreateMsgType(ErrorMessageInitList, 2)
	DoneMessage          = CreateMsgType(DoneMessageInitList, 3)
)

const (
	ControlMessageType = 16
	ErrorMessageType   = 2
	DoneMessageType    = 3
)

/*
Global map witch maps family_id to MessageType
Used for decoding/deserializing of incoming nl msgs
*/

var (
	Family2MT       = make(map[uint16]*MessageType)
	LookupOnStartup = make(map[string][]AttrListTuple)
	MT2Family       = make(map[string]uint16)
	ATLName2ATL     = make(map[string][]AttrTuple)
)

/* Interface for abstraction over native generic netlink types */
type SerDes interface {
	Serialize() ([]byte, error)
	Deserialize([]byte) error
	Val()
}

type U8Type uint8

func (u8 *U8Type) Serialize() ([]byte, error) {
	writer := new(bytes.Buffer)
	err := binary.Write(writer, binary.LittleEndian, u8)
	if err != nil {
		return nil, err
	}
	return writer.Bytes(), nil
}

func (u8 *U8Type) Deserialize(buf []byte) error {
	reader := bytes.NewReader(buf)
	err := binary.Read(reader, binary.LittleEndian, u8)
	if err != nil {
		return fmt.Errorf("error druing deserialization, %v\n", err)
	}
	return nil
}

func (u8 *U8Type) Val() {
	fmt.Println(uint8(*u8))
}

type U16Type uint16

func (u16 *U16Type) Serialize() ([]byte, error) {
	writer := new(bytes.Buffer)
	err := binary.Write(writer, binary.LittleEndian, u16)
	if err != nil {
		return nil, err
	}
	return writer.Bytes(), nil
}
func (u16 *U16Type) Deserialize(buf []byte) error {
	reader := bytes.NewReader(buf)
	err := binary.Read(reader, binary.LittleEndian, u16)
	if err != nil {
		return fmt.Errorf("error druing deserialization, %v\n", err)
	}
	return nil

}

func (u16 *U16Type) Val() {
	fmt.Println(uint16(*u16))
}

type U32Type uint32

func (u32 *U32Type) Serialize() ([]byte, error) {
	writer := new(bytes.Buffer)
	err := binary.Write(writer, binary.LittleEndian, u32)
	if err != nil {
		return nil, err
	}
	return writer.Bytes(), nil

}

func (u32 *U32Type) Deserialize(buf []byte) error {
	reader := bytes.NewReader(buf)
	err := binary.Read(reader, binary.LittleEndian, u32)
	if err != nil {
		return fmt.Errorf("error druing deserialization, %v\n", err)
	}
	return nil

}

func (u32 *U32Type) Val() {
	fmt.Println(uint32(*u32))
}

type I32Type int32

func (i32 *I32Type) Serialize() ([]byte, error) {
	writer := new(bytes.Buffer)
	err := binary.Write(writer, binary.LittleEndian, i32)
	if err != nil {
		return nil, err
	}
	return writer.Bytes(), nil
}

func (i32 *I32Type) Deserialize(buf []byte) error {
	reader := bytes.NewReader(buf)
	err := binary.Read(reader, binary.LittleEndian, i32)
	if err != nil {
		return fmt.Errorf("error druing deserialization, %v\n", err)
	}
	return nil

}

func (i32 *I32Type) Val() {
	fmt.Println(int32(*i32))
}

type U64Type uint64

func (u64 *U64Type) Serialize() ([]byte, error) {
	writer := new(bytes.Buffer)
	err := binary.Write(writer, binary.LittleEndian, u64)
	if err != nil {
		return nil, err
	}
	return writer.Bytes(), nil
}

func (u64 *U64Type) Deserialize(buf []byte) error {
	reader := bytes.NewReader(buf)
	err := binary.Read(reader, binary.LittleEndian, u64)
	if err != nil {
		return fmt.Errorf("error druing deserialization, %v\n", err)
	}
	return nil

}

func (u64 *U64Type) Val() {
	fmt.Println(uint64(*u64))
}

type Net16Type uint16

func (n16 *Net16Type) Serialize() ([]byte, error) {
	writer := new(bytes.Buffer)
	err := binary.Write(writer, binary.BigEndian, n16)
	if err != nil {
		return nil, err
	}
	return writer.Bytes(), nil
}

func (n16 *Net16Type) Deserialize(buf []byte) error {
	reader := bytes.NewReader(buf)
	err := binary.Read(reader, binary.BigEndian, n16)
	if err != nil {
		return fmt.Errorf("error druing deserialization, %v\n", err)
	}
	return nil

}

func (n16 *Net16Type) Val() {
	fmt.Println(uint16(*n16))
}

type Net32Type uint32

func (n32 *Net32Type) Serialize() ([]byte, error) {
	writer := new(bytes.Buffer)
	err := binary.Write(writer, binary.BigEndian, n32)
	if err != nil {
		return nil, err
	}
	return writer.Bytes(), nil
}

func (n32 *Net32Type) Deserialize(buf []byte) error {
	reader := bytes.NewReader(buf)
	err := binary.Read(reader, binary.BigEndian, n32)
	if err != nil {
		return fmt.Errorf("error druing deserialization, %v\n", err)
	}
	return nil

}

func (n32 *Net32Type) Val() {
	fmt.Println(uint32(*n32))
}

type NulStringType string

func (ns NulStringType) Serialize() ([]byte, error) {
	return append([]byte(ns), 0), nil
}

func (ns *NulStringType) Deserialize(buf []byte) error {
	if buf[len(buf)-1] != 0 {
		return fmt.Errorf("non 0 terminated string\n")
	}
	s := string(buf[:len(buf)-1])
	*ns = NulStringType(s)
	return nil

}

func (ns *NulStringType) Val() {
	fmt.Println(string(*ns))
}

type IgnoreType bool

func (it *IgnoreType) Serialize() ([]byte, error) {
	return nil, nil
}

func (it *IgnoreType) Deserialize(buf []byte) error {
	return nil
}

func (it *IgnoreType) Val() {
	fmt.Println("ignore type")
}

type BinaryType []byte

func (bt *BinaryType) Serialize() ([]byte, error) {
	byte_slice := make([]byte, 0)
	byte_slice = append(byte_slice, []byte(*bt)...)
	return byte_slice, nil
}

func (bt *BinaryType) Deserialize(buf []byte) error {
	byte_slice := make([]byte, 0)
	byte_slice = append(byte_slice, buf...)
	*bt = BinaryType(byte_slice)
	return nil
}

func (bt *BinaryType) Val() {
	fmt.Println(*bt)
}

/*
This struct has been used for describing and constructing netlink's
attributes lists.
*/
type AttrTuple struct {
	Name string
	Type string
}

type AttrHdr struct {
	Len uint16
	Num uint16
}

/*
Struct describes how we encode (which netlink's msg type number coresponds to which type)
Amap contains dict of SerDes types, which we can serialize or where we will deserialize incoming
msg.
*/
type AttrListType struct {
	Key2name map[int]string
	Name2key map[string]int
	Key2Type map[int]string
	Amap     map[string]SerDes
}

/*
Routine which helps us to create global attributelists definition dict.
*/
func CreateAttrListDefinition(listName string, atl []AttrTuple) []AttrTuple {
	ATLName2ATL[listName] = atl
	return atl
}

func CreateAttrListType(attrListMap []AttrTuple) AttrListType {
	al := new(AttrListType)
	al.Key2name = make(map[int]string)
	al.Name2key = make(map[string]int)
	al.Key2Type = make(map[int]string)
	al.Amap = make(map[string]SerDes)
	for i, attr := range attrListMap {
		key := i + 1
		al.Key2name[key] = attr.Name
		al.Key2Type[key] = attr.Type
		al.Name2key[attr.Name] = key
	}
	return *al
}

func (al *AttrListType) Set(amap map[string]SerDes) {
	al.Amap = amap
}

func (al *AttrListType) Serialize() ([]byte, error) {
	buf := make([]byte, 0)
	pad := make([]byte, 4)
	for attrType, attrData := range al.Amap {
		if attrNum, exists := al.Name2key[attrType]; !exists {
			return nil, fmt.Errorf("err. amap and attrList are incompatible:%#v\n%#v\n", al.Name2key, attrType)
		} else {
			data, err := attrData.Serialize()
			if err != nil {
				return nil, err
			}
			attrLen := AttrHdr{Len: uint16(len(data) + 4), Num: uint16(attrNum)}
			attrBuf := new(bytes.Buffer)
			err = binary.Write(attrBuf, binary.LittleEndian, attrLen)
			if err != nil {
				return nil, err
			}
			/*
				TODO(tehnerd): lots of padding hack's translated from gnlpy as is.
				prob one day gonna read more about it.
			*/
			buf = append(buf, attrBuf.Bytes()...)
			buf = append(buf, data...)
			padLen := (4 - (len(data) % 4)) & 0x3
			if padLen > 4 {
				return nil, fmt.Errorf("wrong pad len calc")
			}
			buf = append(buf, pad[:padLen]...)
		}
	}
	return buf, nil
}

/*
Routine which helps to deserialize. used in AttrList Deserialize()
*/
func DeserializeSerDes(serdesType string, list []byte) (SerDes, error) {
	switch serdesType {
	case "U8Type":
		attr := new(U8Type)
		err := attr.Deserialize(list)
		if err != nil {
			return nil, err
		}
		return attr, nil
	case "U16Type":
		attr := new(U16Type)
		err := attr.Deserialize(list)
		if err != nil {
			return nil, err
		}
		return attr, nil
	case "U32Type":
		attr := new(U32Type)
		err := attr.Deserialize(list)
		if err != nil {
			return nil, err
		}
		return attr, nil
	case "U64Type":
		attr := new(U64Type)
		err := attr.Deserialize(list)
		if err != nil {
			return nil, err
		}
		return attr, nil
	case "I32Type":
		attr := new(I32Type)
		err := attr.Deserialize(list)
		if err != nil {
			return nil, err
		}
		return attr, nil
	case "Net16Type":
		attr := new(Net16Type)
		err := attr.Deserialize(list)
		if err != nil {
			return nil, err
		}
		return attr, nil
	case "Net32Type":
		attr := new(Net32Type)
		err := attr.Deserialize(list)
		if err != nil {
			return nil, err
		}
		return attr, nil
	case "NulStringType":
		attr := new(NulStringType)
		err := attr.Deserialize(list)
		if err != nil {
			return nil, err
		}
		return attr, nil
	case "IgnoreType":
		attr := new(IgnoreType)
		return attr, nil
	case "BinaryType":
		attr := new(BinaryType)
		//binary always return nil, no point to check err != nil
		attr.Deserialize(list)
		return attr, nil
	/*
		XXX(tehnerd): dangerous assumption that we either have basic types (above) or it's
		a nested attribute's list. havent tested in prod yet
	*/
	default:
		atl, exists := ATLName2ATL[serdesType]
		if !exists {
			return nil, fmt.Errorf("serdes doesnt exists. type: %v\n", serdesType)
		}
		attr := CreateAttrListType(atl)
		err := attr.Deserialize(list)
		if err != nil {
			return nil, err
		}
		return &attr, nil
	}
	return nil, nil
}

func (al *AttrListType) Deserialize(list []byte) error {
	al.Amap = make(map[string]SerDes)
	var attrHdr AttrHdr
	for len(list) > 0 {
		err := binary.Read(bytes.NewReader(list), binary.LittleEndian, &attrHdr)
		if err != nil {
			return fmt.Errorf("cant read attr header for deserialization: %v\n", err)
		}
		//XXX(tehnerd): again fb's hacks
		attrHdr.Len = attrHdr.Len & 0x7fff
		//TODO(tehnerd): no support for "RecursiveSelf" as for now
		fieldType, exists := al.Key2Type[int(attrHdr.Num)]
		if !exists {
			list = list[(int(attrHdr.Len+3) & (^3)):]
			//TODO(tehnerd): hack. had panics on ipvs's PE_NAME
			continue
		}
		fieldName := al.Key2name[int(attrHdr.Num)]
		al.Amap[fieldName], err = DeserializeSerDes(fieldType, list[4:attrHdr.Len])
		if err != nil {
			return err
		}
		list = list[(int(attrHdr.Len+3) & (^3)):]
	}
	return nil
}

func (al *AttrListType) Val() {
	for k, v := range al.Amap {
		fmt.Println(k)
		v.Val()
	}
}

/*
Struct, which describes how generic netlink message for particular family
could looks like
*/
type MessageType struct {
	Name2key         map[string]int
	Key2name         map[int]string
	Key2attrListType map[int]AttrListType
	Family           uint16
}

type AttrListTuple struct {
	Name     string
	AttrList AttrListType
}

/*
Routine, which helps us to create global dict of msg types
*/
func CreateMsgType(alist []AttrListTuple, familyId uint16) MessageType {
	if v, exists := Family2MT[familyId]; exists {
		return *v
	}
	var mt MessageType
	mt.InitMessageType(alist, familyId)
	Family2MT[familyId] = &mt
	return mt
}

/*
Routine, which helps us to resolve family's name to ID on startup
*/
func LookupTypeOnStartup(alist []AttrListTuple, familyName string) {
	LookupOnStartup[familyName] = alist
}

func (mt *MessageType) InitMessageType(alist []AttrListTuple, familyId uint16) {
	mt.Name2key = make(map[string]int)
	mt.Key2name = make(map[int]string)
	mt.Key2attrListType = make(map[int]AttrListType)
	mt.Family = familyId
	for i, attrTyple := range alist {
		key := i + 1
		mt.Name2key[attrTyple.Name] = key
		mt.Key2name[key] = attrTyple.Name
		mt.Key2attrListType[key] = attrTyple.AttrList
	}

}

//GNL - generic netlink msg. NL msg contains NLmsgHdr + GNLMsg
type GNLMsgHdr struct {
	Cmnd    uint8
	Version uint8
}

type GNLMessage struct {
	Hdr     GNLMsgHdr
	AttrMap map[string]SerDes
	Family  uint16
	Flags   uint16
	MT      *MessageType
}

func (msg *GNLMessage) Init(hdr GNLMsgHdr, amap map[string]SerDes,
	family, flags uint16) {
	msg.Hdr = hdr
	msg.AttrMap = amap
	msg.Family = family
	msg.Flags = flags
}

func (mt *MessageType) InitGNLMessageStr(cmnd string, flags uint16) (GNLMessage, error) {
	var gnlMsg GNLMessage
	cmndId, exists := mt.Name2key[cmnd]
	if !exists {
		return gnlMsg, fmt.Errorf("cmnd with name %s doesnt exists\n", cmnd)
	}
	amap := make(map[string]SerDes)
	gnlMsg.Init(GNLMsgHdr{Cmnd: uint8(cmndId), Version: 1},
		amap,
		mt.Family,
		flags)
	gnlMsg.MT = mt
	return gnlMsg, nil
}

func (msg *GNLMessage) GetAttrList(name string) SerDes {
	return msg.AttrMap[name]
}

func (msg *GNLMessage) SetAttrList(name string, val SerDes) {
	msg.AttrMap[name] = val
}

func (mt *MessageType) SerializeGNLMsg(msg GNLMessage) ([]byte, error) {
	sMsg := make([]byte, 0)
	buf := new(bytes.Buffer)
	err := binary.Write(buf, binary.LittleEndian, msg.Hdr)
	if err != nil {
		return nil, fmt.Errorf("cant serialize gnlmsg hdr: %v\n", err)
	}
	sMsg = append(sMsg, buf.Bytes()...)
	//padding
	sMsg = append(sMsg, []byte{0, 0}...)
	if v, exists := mt.Key2attrListType[int(msg.Hdr.Cmnd)]; !exists {
		return nil, fmt.Errorf("no existing cmnd in %#v\n", msg.Hdr)
	} else {
		v.Set(msg.AttrMap)
		vSer, err := v.Serialize()
		if err != nil {
			return nil, err
		}
		sMsg = append(sMsg, vSer...)
	}
	return sMsg, nil
}

func (mt *MessageType) DeserializeGNLMsg(sMsg []byte) (GNLMessage, error) {
	var msgHdr GNLMsgHdr
	err := binary.Read(bytes.NewReader(sMsg), binary.LittleEndian, &msgHdr)
	if err != nil {
		return GNLMessage{}, fmt.Errorf("cant read(deserialize) msg hdr: %v\n", err)
	}
	v, exists := mt.Key2attrListType[int(msgHdr.Cmnd)]
	if !exists {
		return GNLMessage{}, fmt.Errorf("no such cmnd in key2attrlist dict: %#v\nkey2attrlist: %#v\n",
			msgHdr, mt.Key2attrListType)
	}
	err = v.Deserialize(sMsg[4:])
	if err != nil {
		return GNLMessage{}, err
	}
	var msg GNLMessage
	msg.Init(msgHdr, v.Amap, 1, ACK_REQUEST)
	return msg, nil
}

type NLMsgHdr struct {
	TotalLen uint32
	Family   uint16
	Flags    uint16
	Seq      uint32
	PortID   uint32
}

func SerializeNLMsg(mt *MessageType, msg GNLMessage, portId, Seq uint32) ([]byte, error) {
	nlMsg := make([]byte, 0)
	payload, err := mt.SerializeGNLMsg(msg)
	if err != nil {
		return nil, err
	}
	nlHdr := NLMsgHdr{
		TotalLen: uint32(len(payload) + 16),
		Family:   msg.Family,
		Flags:    msg.Flags,
		Seq:      Seq,
		PortID:   portId}
	buf := new(bytes.Buffer)
	err = binary.Write(buf, binary.LittleEndian, nlHdr)
	if err != nil {
		return nil, err
	}
	nlMsg = append(nlMsg, buf.Bytes()...)
	nlMsg = append(nlMsg, payload...)
	return nlMsg, nil
}

func DeserializeNLMsg(sMsg []byte) (GNLMessage, []byte, error) {
	var nlHdr NLMsgHdr
	err := binary.Read(bytes.NewReader(sMsg), binary.LittleEndian, &nlHdr)
	if err != nil {
		return GNLMessage{}, nil, err
	}
	mt, exists := Family2MT[nlHdr.Family]
	if !exists {
		return GNLMessage{}, nil,
			fmt.Errorf("msg with such family doesn exist in mType dict: %#v\n", nlHdr)
	}
	if nlHdr.Family == ErrorMessageType {
		var ErrorCode int32
		binary.Read(bytes.NewReader(sMsg[16:]), binary.LittleEndian, &ErrorCode)
		if ErrorCode != 0 {
			/*
				TODO(tehnerd): we could return which msg was the reason of error
				if len(sMsg) > 20 {
					_, _, err := DeserializeNLMsg(sMsg[20:])
				}
			*/
			return GNLMessage{}, nil, fmt.Errorf("Error! errorcode is: %d\n", -ErrorCode)
		} else {
			return GNLMessage{}, nil, nil
		}
	} else if nlHdr.Family == DoneMessageType {
		var msg GNLMessage
		msg.Family = nlHdr.Family
		msg.Flags = nlHdr.Flags
		return msg, nil, nil
	}
	msg, err := mt.DeserializeGNLMsg(sMsg[16:])
	if err != nil {
		return GNLMessage{}, nil, err
	}
	msg.Family = nlHdr.Family
	msg.Flags = nlHdr.Flags
	return msg, sMsg[nlHdr.TotalLen:], nil

}

type NLSocket struct {
	Sd      int
	Seq     uint32
	PortID  uint32
	Lock    *sync.Mutex
	Verbose bool
}

func (nlSock *NLSocket) Init() error {
	//16 - NETLINK_GENERIC
	sd, err := syscall.Socket(syscall.AF_NETLINK, syscall.SOCK_DGRAM, NETLINK_GENERIC)
	if err != nil {
		return fmt.Errorf("cant create netlink socket %v\n", err)
	}
	pid := uint32(syscall.Getpid())
	sa := &syscall.SockaddrNetlink{
		Pid:    pid,
		Groups: 0,
		Family: syscall.AF_NETLINK}
	if err = syscall.Bind(sd, sa); err != nil {
		return fmt.Errorf("cant bind to netlink socket: %v\n", err)
	}
	nlSock.Lock = new(sync.Mutex)
	nlSock.Sd = sd
	nlSock.Seq = 0
	nlSock.PortID = pid
	for k, v := range LookupOnStartup {
		familyId, err := nlSock.ResolveFamily(NulStringType(k))
		if err != nil {
			return err
		}
		CreateMsgType(v, uint16(*familyId))
		MT2Family[k] = uint16(*familyId)
	}
	return nil
}

func (nlSock *NLSocket) Close() {
	syscall.Close(nlSock.Sd)
}

func (nlSock *NLSocket) ResolveFamily(family NulStringType) (*U16Type, error) {
	gnlMsg, err := CtrlMessage.InitGNLMessageStr("GETFAMILY", REQUEST)
	if err != nil {
		return nil, err
	}
	gnlMsg.SetAttrList("FAMILY_NAME", &family)
	reply, err := nlSock.Query(gnlMsg)
	if err != nil {
		return nil, err
	}
	familyId := reply[0].GetAttrList("FAMILY_ID")
	//we are going to panic if it's  not U16Typ
	familyIdPtr, ok := familyId.(*U16Type)
	if !ok {
		return nil, fmt.Errorf("cant convert familyId to U16Type")
	}
	return familyIdPtr, nil
}

func (nlSock *NLSocket) Query(msg GNLMessage) ([]GNLMessage, error) {
	nlSock.Lock.Lock()
	defer nlSock.Lock.Unlock()
	nlSock.send(msg)
	resp, err := nlSock.recv()
	if err != nil {
		return nil, err
	}
	return resp, nil
}

func (nlSock *NLSocket) send(msg GNLMessage) error {
	data, err := SerializeNLMsg(msg.MT, msg, nlSock.PortID, nlSock.Seq)
	if err != nil {
		return err
	}
	nlSock.Seq += 1
	lsa := &syscall.SockaddrNetlink{Family: syscall.AF_NETLINK}
	err = syscall.Sendto(nlSock.Sd, data, 0, lsa)
	if err != nil {
		return err
	}
	return nil
}

func (nlSock *NLSocket) recv() ([]GNLMessage, error) {
	buff := make([]byte, 16384)
	var msgsList []GNLMessage
	for {
		n, _, err := syscall.Recvfrom(nlSock.Sd, buff, 0)
		if err != nil {
			return nil, err
		}
		resp := buff[:n]
		for len(resp) > 0 {
			rmsg, data, err := DeserializeNLMsg(resp)
			if err != nil {
				return nil, err
			}
			if len(msgsList) == 0 && rmsg.Flags&0x2 == 0 {
				return []GNLMessage{rmsg}, nil
			} else if rmsg.Family == DoneMessageType {
				return msgsList, nil
			}
			msgsList = append(msgsList, rmsg)
			resp = data
		}
	}
	return msgsList, nil
}

func (nlSock *NLSocket) Execute(msg GNLMessage) error {
	nlSock.Lock.Lock()
	defer nlSock.Lock.Unlock()
	err := nlSock.send(msg)
	if err != nil {
		return err
	}
	resp, err := nlSock.recv()
	if err != nil {
		return err
	}
	if len(resp) != 1 {
		return fmt.Errorf("we dont expect more than one msg in response\n")
	}
	if resp[0].Family == ErrorMessageType {
		return fmt.Errorf("error in response of execution\n")
	}
	return nil
}

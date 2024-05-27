package netlink

import (
	"fmt"
	"net"
	"strings"
	"syscall"

	"github.com/vishvananda/netlink/nl"
	"golang.org/x/sys/unix"
)

// DevlinkDevEswitchAttr represents device's eswitch attributes
type DevlinkDevEswitchAttr struct {
	Mode       string
	InlineMode string
	EncapMode  string
}

// DevlinkDevAttrs represents device attributes
type DevlinkDevAttrs struct {
	Eswitch DevlinkDevEswitchAttr
}

// DevlinkDevice represents device and its attributes
type DevlinkDevice struct {
	BusName    string
	DeviceName string
	Attrs      DevlinkDevAttrs
}

// DevlinkPortFn represents port function and its attributes
type DevlinkPortFn struct {
	HwAddr  net.HardwareAddr
	State   uint8
	OpState uint8
}

// DevlinkPortFnSetAttrs represents attributes to set
type DevlinkPortFnSetAttrs struct {
	FnAttrs     DevlinkPortFn
	HwAddrValid bool
	StateValid  bool
}

// DevlinkPort represents port and its attributes
type DevlinkPort struct {
	BusName        string
	DeviceName     string
	PortIndex      uint32
	PortType       uint16
	NetdeviceName  string
	NetdevIfIndex  uint32
	RdmaDeviceName string
	PortFlavour    uint16
	Fn             *DevlinkPortFn
}

type DevLinkPortAddAttrs struct {
	Controller      uint32
	SfNumber        uint32
	PortIndex       uint32
	PfNumber        uint16
	SfNumberValid   bool
	PortIndexValid  bool
	ControllerValid bool
}

// DevlinkDeviceInfo represents devlink info
type DevlinkDeviceInfo struct {
	Driver         string
	SerialNumber   string
	BoardID        string
	FwApp          string
	FwAppBoundleID string
	FwAppName      string
	FwBoundleID    string
	FwMgmt         string
	FwMgmtAPI      string
	FwMgmtBuild    string
	FwNetlist      string
	FwNetlistBuild string
	FwPsidAPI      string
	FwUndi         string
}

func parseDevLinkDeviceList(msgs [][]byte) ([]*DevlinkDevice, error) {
	devices := make([]*DevlinkDevice, 0, len(msgs))
	for _, m := range msgs {
		attrs, err := nl.ParseRouteAttr(m[nl.SizeofGenlmsg:])
		if err != nil {
			return nil, err
		}
		dev := &DevlinkDevice{}
		if err = dev.parseAttributes(attrs); err != nil {
			return nil, err
		}
		devices = append(devices, dev)
	}
	return devices, nil
}

func eswitchStringToMode(modeName string) (uint16, error) {
	if modeName == "legacy" {
		return nl.DEVLINK_ESWITCH_MODE_LEGACY, nil
	} else if modeName == "switchdev" {
		return nl.DEVLINK_ESWITCH_MODE_SWITCHDEV, nil
	} else {
		return 0xffff, fmt.Errorf("invalid switchdev mode")
	}
}

func parseEswitchMode(mode uint16) string {
	var eswitchMode = map[uint16]string{
		nl.DEVLINK_ESWITCH_MODE_LEGACY:    "legacy",
		nl.DEVLINK_ESWITCH_MODE_SWITCHDEV: "switchdev",
	}
	if eswitchMode[mode] == "" {
		return "unknown"
	} else {
		return eswitchMode[mode]
	}
}

func parseEswitchInlineMode(inlinemode uint8) string {
	var eswitchInlineMode = map[uint8]string{
		nl.DEVLINK_ESWITCH_INLINE_MODE_NONE:      "none",
		nl.DEVLINK_ESWITCH_INLINE_MODE_LINK:      "link",
		nl.DEVLINK_ESWITCH_INLINE_MODE_NETWORK:   "network",
		nl.DEVLINK_ESWITCH_INLINE_MODE_TRANSPORT: "transport",
	}
	if eswitchInlineMode[inlinemode] == "" {
		return "unknown"
	} else {
		return eswitchInlineMode[inlinemode]
	}
}

func parseEswitchEncapMode(encapmode uint8) string {
	var eswitchEncapMode = map[uint8]string{
		nl.DEVLINK_ESWITCH_ENCAP_MODE_NONE:  "disable",
		nl.DEVLINK_ESWITCH_ENCAP_MODE_BASIC: "enable",
	}
	if eswitchEncapMode[encapmode] == "" {
		return "unknown"
	} else {
		return eswitchEncapMode[encapmode]
	}
}

func (d *DevlinkDevice) parseAttributes(attrs []syscall.NetlinkRouteAttr) error {
	for _, a := range attrs {
		switch a.Attr.Type {
		case nl.DEVLINK_ATTR_BUS_NAME:
			d.BusName = string(a.Value[:len(a.Value)-1])
		case nl.DEVLINK_ATTR_DEV_NAME:
			d.DeviceName = string(a.Value[:len(a.Value)-1])
		case nl.DEVLINK_ATTR_ESWITCH_MODE:
			d.Attrs.Eswitch.Mode = parseEswitchMode(native.Uint16(a.Value))
		case nl.DEVLINK_ATTR_ESWITCH_INLINE_MODE:
			d.Attrs.Eswitch.InlineMode = parseEswitchInlineMode(uint8(a.Value[0]))
		case nl.DEVLINK_ATTR_ESWITCH_ENCAP_MODE:
			d.Attrs.Eswitch.EncapMode = parseEswitchEncapMode(uint8(a.Value[0]))
		}
	}
	return nil
}

func (dev *DevlinkDevice) parseEswitchAttrs(msgs [][]byte) {
	m := msgs[0]
	attrs, err := nl.ParseRouteAttr(m[nl.SizeofGenlmsg:])
	if err != nil {
		return
	}
	dev.parseAttributes(attrs)
}

func (h *Handle) getEswitchAttrs(family *GenlFamily, dev *DevlinkDevice) {
	msg := &nl.Genlmsg{
		Command: nl.DEVLINK_CMD_ESWITCH_GET,
		Version: nl.GENL_DEVLINK_VERSION,
	}
	req := h.newNetlinkRequest(int(family.ID), unix.NLM_F_REQUEST|unix.NLM_F_ACK)
	req.AddData(msg)

	b := make([]byte, len(dev.BusName)+1)
	copy(b, dev.BusName)
	data := nl.NewRtAttr(nl.DEVLINK_ATTR_BUS_NAME, b)
	req.AddData(data)

	b = make([]byte, len(dev.DeviceName)+1)
	copy(b, dev.DeviceName)
	data = nl.NewRtAttr(nl.DEVLINK_ATTR_DEV_NAME, b)
	req.AddData(data)

	msgs, err := req.Execute(unix.NETLINK_GENERIC, 0)
	if err != nil {
		return
	}
	dev.parseEswitchAttrs(msgs)
}

// DevLinkGetDeviceList provides a pointer to devlink devices and nil error,
// otherwise returns an error code.
func (h *Handle) DevLinkGetDeviceList() ([]*DevlinkDevice, error) {
	f, err := h.GenlFamilyGet(nl.GENL_DEVLINK_NAME)
	if err != nil {
		return nil, err
	}
	msg := &nl.Genlmsg{
		Command: nl.DEVLINK_CMD_GET,
		Version: nl.GENL_DEVLINK_VERSION,
	}
	req := h.newNetlinkRequest(int(f.ID),
		unix.NLM_F_REQUEST|unix.NLM_F_ACK|unix.NLM_F_DUMP)
	req.AddData(msg)
	msgs, err := req.Execute(unix.NETLINK_GENERIC, 0)
	if err != nil {
		return nil, err
	}
	devices, err := parseDevLinkDeviceList(msgs)
	if err != nil {
		return nil, err
	}
	for _, d := range devices {
		h.getEswitchAttrs(f, d)
	}
	return devices, nil
}

// DevLinkGetDeviceList provides a pointer to devlink devices and nil error,
// otherwise returns an error code.
func DevLinkGetDeviceList() ([]*DevlinkDevice, error) {
	return pkgHandle.DevLinkGetDeviceList()
}

func parseDevlinkDevice(msgs [][]byte) (*DevlinkDevice, error) {
	m := msgs[0]
	attrs, err := nl.ParseRouteAttr(m[nl.SizeofGenlmsg:])
	if err != nil {
		return nil, err
	}
	dev := &DevlinkDevice{}
	if err = dev.parseAttributes(attrs); err != nil {
		return nil, err
	}
	return dev, nil
}

func (h *Handle) createCmdReq(cmd uint8, bus string, device string) (*GenlFamily, *nl.NetlinkRequest, error) {
	f, err := h.GenlFamilyGet(nl.GENL_DEVLINK_NAME)
	if err != nil {
		return nil, nil, err
	}

	msg := &nl.Genlmsg{
		Command: cmd,
		Version: nl.GENL_DEVLINK_VERSION,
	}
	req := h.newNetlinkRequest(int(f.ID),
		unix.NLM_F_REQUEST|unix.NLM_F_ACK)
	req.AddData(msg)

	b := make([]byte, len(bus)+1)
	copy(b, bus)
	data := nl.NewRtAttr(nl.DEVLINK_ATTR_BUS_NAME, b)
	req.AddData(data)

	b = make([]byte, len(device)+1)
	copy(b, device)
	data = nl.NewRtAttr(nl.DEVLINK_ATTR_DEV_NAME, b)
	req.AddData(data)

	return f, req, nil
}

// DevlinkGetDeviceByName provides a pointer to devlink device and nil error,
// otherwise returns an error code.
func (h *Handle) DevLinkGetDeviceByName(Bus string, Device string) (*DevlinkDevice, error) {
	f, req, err := h.createCmdReq(nl.DEVLINK_CMD_GET, Bus, Device)
	if err != nil {
		return nil, err
	}

	respmsg, err := req.Execute(unix.NETLINK_GENERIC, 0)
	if err != nil {
		return nil, err
	}
	dev, err := parseDevlinkDevice(respmsg)
	if err == nil {
		h.getEswitchAttrs(f, dev)
	}
	return dev, err
}

// DevlinkGetDeviceByName provides a pointer to devlink device and nil error,
// otherwise returns an error code.
func DevLinkGetDeviceByName(Bus string, Device string) (*DevlinkDevice, error) {
	return pkgHandle.DevLinkGetDeviceByName(Bus, Device)
}

// DevLinkSetEswitchMode sets eswitch mode if able to set successfully or
// returns an error code.
// Equivalent to: `devlink dev eswitch set $dev mode switchdev`
// Equivalent to: `devlink dev eswitch set $dev mode legacy`
func (h *Handle) DevLinkSetEswitchMode(Dev *DevlinkDevice, NewMode string) error {
	mode, err := eswitchStringToMode(NewMode)
	if err != nil {
		return err
	}

	_, req, err := h.createCmdReq(nl.DEVLINK_CMD_ESWITCH_SET, Dev.BusName, Dev.DeviceName)
	if err != nil {
		return err
	}

	req.AddData(nl.NewRtAttr(nl.DEVLINK_ATTR_ESWITCH_MODE, nl.Uint16Attr(mode)))

	_, err = req.Execute(unix.NETLINK_GENERIC, 0)
	return err
}

// DevLinkSetEswitchMode sets eswitch mode if able to set successfully or
// returns an error code.
// Equivalent to: `devlink dev eswitch set $dev mode switchdev`
// Equivalent to: `devlink dev eswitch set $dev mode legacy`
func DevLinkSetEswitchMode(Dev *DevlinkDevice, NewMode string) error {
	return pkgHandle.DevLinkSetEswitchMode(Dev, NewMode)
}

func (port *DevlinkPort) parseAttributes(attrs []syscall.NetlinkRouteAttr) error {
	for _, a := range attrs {
		switch a.Attr.Type {
		case nl.DEVLINK_ATTR_BUS_NAME:
			port.BusName = string(a.Value[:len(a.Value)-1])
		case nl.DEVLINK_ATTR_DEV_NAME:
			port.DeviceName = string(a.Value[:len(a.Value)-1])
		case nl.DEVLINK_ATTR_PORT_INDEX:
			port.PortIndex = native.Uint32(a.Value)
		case nl.DEVLINK_ATTR_PORT_TYPE:
			port.PortType = native.Uint16(a.Value)
		case nl.DEVLINK_ATTR_PORT_NETDEV_NAME:
			port.NetdeviceName = string(a.Value[:len(a.Value)-1])
		case nl.DEVLINK_ATTR_PORT_NETDEV_IFINDEX:
			port.NetdevIfIndex = native.Uint32(a.Value)
		case nl.DEVLINK_ATTR_PORT_IBDEV_NAME:
			port.RdmaDeviceName = string(a.Value[:len(a.Value)-1])
		case nl.DEVLINK_ATTR_PORT_FLAVOUR:
			port.PortFlavour = native.Uint16(a.Value)
		case nl.DEVLINK_ATTR_PORT_FUNCTION:
			port.Fn = &DevlinkPortFn{}
			for nested := range nl.ParseAttributes(a.Value) {
				switch nested.Type {
				case nl.DEVLINK_PORT_FUNCTION_ATTR_HW_ADDR:
					port.Fn.HwAddr = nested.Value[:]
				case nl.DEVLINK_PORT_FN_ATTR_STATE:
					port.Fn.State = uint8(nested.Value[0])
				case nl.DEVLINK_PORT_FN_ATTR_OPSTATE:
					port.Fn.OpState = uint8(nested.Value[0])
				}
			}
		}
	}
	return nil
}

func parseDevLinkAllPortList(msgs [][]byte) ([]*DevlinkPort, error) {
	ports := make([]*DevlinkPort, 0, len(msgs))
	for _, m := range msgs {
		attrs, err := nl.ParseRouteAttr(m[nl.SizeofGenlmsg:])
		if err != nil {
			return nil, err
		}
		port := &DevlinkPort{}
		if err = port.parseAttributes(attrs); err != nil {
			return nil, err
		}
		ports = append(ports, port)
	}
	return ports, nil
}

// DevLinkGetPortList provides a pointer to devlink ports and nil error,
// otherwise returns an error code.
func (h *Handle) DevLinkGetAllPortList() ([]*DevlinkPort, error) {
	f, err := h.GenlFamilyGet(nl.GENL_DEVLINK_NAME)
	if err != nil {
		return nil, err
	}
	msg := &nl.Genlmsg{
		Command: nl.DEVLINK_CMD_PORT_GET,
		Version: nl.GENL_DEVLINK_VERSION,
	}
	req := h.newNetlinkRequest(int(f.ID),
		unix.NLM_F_REQUEST|unix.NLM_F_ACK|unix.NLM_F_DUMP)
	req.AddData(msg)
	msgs, err := req.Execute(unix.NETLINK_GENERIC, 0)
	if err != nil {
		return nil, err
	}
	ports, err := parseDevLinkAllPortList(msgs)
	if err != nil {
		return nil, err
	}
	return ports, nil
}

// DevLinkGetPortList provides a pointer to devlink ports and nil error,
// otherwise returns an error code.
func DevLinkGetAllPortList() ([]*DevlinkPort, error) {
	return pkgHandle.DevLinkGetAllPortList()
}

func parseDevlinkPortMsg(msgs [][]byte) (*DevlinkPort, error) {
	m := msgs[0]
	attrs, err := nl.ParseRouteAttr(m[nl.SizeofGenlmsg:])
	if err != nil {
		return nil, err
	}
	port := &DevlinkPort{}
	if err = port.parseAttributes(attrs); err != nil {
		return nil, err
	}
	return port, nil
}

// DevLinkGetPortByIndexprovides a pointer to devlink device and nil error,
// otherwise returns an error code.
func (h *Handle) DevLinkGetPortByIndex(Bus string, Device string, PortIndex uint32) (*DevlinkPort, error) {

	_, req, err := h.createCmdReq(nl.DEVLINK_CMD_PORT_GET, Bus, Device)
	if err != nil {
		return nil, err
	}

	req.AddData(nl.NewRtAttr(nl.DEVLINK_ATTR_PORT_INDEX, nl.Uint32Attr(PortIndex)))

	respmsg, err := req.Execute(unix.NETLINK_GENERIC, 0)
	if err != nil {
		return nil, err
	}
	port, err := parseDevlinkPortMsg(respmsg)
	return port, err
}

// DevLinkGetPortByIndex provides a pointer to devlink portand nil error,
// otherwise returns an error code.
func DevLinkGetPortByIndex(Bus string, Device string, PortIndex uint32) (*DevlinkPort, error) {
	return pkgHandle.DevLinkGetPortByIndex(Bus, Device, PortIndex)
}

// DevLinkPortAdd adds a devlink port and returns a port on success
// otherwise returns nil port and an error code.
func (h *Handle) DevLinkPortAdd(Bus string, Device string, Flavour uint16, Attrs DevLinkPortAddAttrs) (*DevlinkPort, error) {
	_, req, err := h.createCmdReq(nl.DEVLINK_CMD_PORT_NEW, Bus, Device)
	if err != nil {
		return nil, err
	}

	req.AddData(nl.NewRtAttr(nl.DEVLINK_ATTR_PORT_FLAVOUR, nl.Uint16Attr(Flavour)))

	req.AddData(nl.NewRtAttr(nl.DEVLINK_ATTR_PORT_PCI_PF_NUMBER, nl.Uint16Attr(Attrs.PfNumber)))
	if Flavour == nl.DEVLINK_PORT_FLAVOUR_PCI_SF && Attrs.SfNumberValid {
		req.AddData(nl.NewRtAttr(nl.DEVLINK_ATTR_PORT_PCI_SF_NUMBER, nl.Uint32Attr(Attrs.SfNumber)))
	}
	if Attrs.PortIndexValid {
		req.AddData(nl.NewRtAttr(nl.DEVLINK_ATTR_PORT_INDEX, nl.Uint32Attr(Attrs.PortIndex)))
	}
	if Attrs.ControllerValid {
		req.AddData(nl.NewRtAttr(nl.DEVLINK_ATTR_PORT_CONTROLLER_NUMBER, nl.Uint32Attr(Attrs.Controller)))
	}
	respmsg, err := req.Execute(unix.NETLINK_GENERIC, 0)
	if err != nil {
		return nil, err
	}
	port, err := parseDevlinkPortMsg(respmsg)
	return port, err
}

// DevLinkPortAdd adds a devlink port and returns a port on success
// otherwise returns nil port and an error code.
func DevLinkPortAdd(Bus string, Device string, Flavour uint16, Attrs DevLinkPortAddAttrs) (*DevlinkPort, error) {
	return pkgHandle.DevLinkPortAdd(Bus, Device, Flavour, Attrs)
}

// DevLinkPortDel deletes a devlink port and returns success or error code.
func (h *Handle) DevLinkPortDel(Bus string, Device string, PortIndex uint32) error {
	_, req, err := h.createCmdReq(nl.DEVLINK_CMD_PORT_DEL, Bus, Device)
	if err != nil {
		return err
	}

	req.AddData(nl.NewRtAttr(nl.DEVLINK_ATTR_PORT_INDEX, nl.Uint32Attr(PortIndex)))
	_, err = req.Execute(unix.NETLINK_GENERIC, 0)
	return err
}

// DevLinkPortDel deletes a devlink port and returns success or error code.
func DevLinkPortDel(Bus string, Device string, PortIndex uint32) error {
	return pkgHandle.DevLinkPortDel(Bus, Device, PortIndex)
}

// DevlinkPortFnSet sets one or more port function attributes specified by the attribute mask.
// It returns 0 on success or error code.
func (h *Handle) DevlinkPortFnSet(Bus string, Device string, PortIndex uint32, FnAttrs DevlinkPortFnSetAttrs) error {
	_, req, err := h.createCmdReq(nl.DEVLINK_CMD_PORT_SET, Bus, Device)
	if err != nil {
		return err
	}

	req.AddData(nl.NewRtAttr(nl.DEVLINK_ATTR_PORT_INDEX, nl.Uint32Attr(PortIndex)))

	fnAttr := nl.NewRtAttr(nl.DEVLINK_ATTR_PORT_FUNCTION|unix.NLA_F_NESTED, nil)

	if FnAttrs.HwAddrValid {
		fnAttr.AddRtAttr(nl.DEVLINK_PORT_FUNCTION_ATTR_HW_ADDR, []byte(FnAttrs.FnAttrs.HwAddr))
	}

	if FnAttrs.StateValid {
		fnAttr.AddRtAttr(nl.DEVLINK_PORT_FN_ATTR_STATE, nl.Uint8Attr(FnAttrs.FnAttrs.State))
	}
	req.AddData(fnAttr)

	_, err = req.Execute(unix.NETLINK_GENERIC, 0)
	return err
}

// DevlinkPortFnSet sets one or more port function attributes specified by the attribute mask.
// It returns 0 on success or error code.
func DevlinkPortFnSet(Bus string, Device string, PortIndex uint32, FnAttrs DevlinkPortFnSetAttrs) error {
	return pkgHandle.DevlinkPortFnSet(Bus, Device, PortIndex, FnAttrs)
}

// devlinkInfoGetter is function that is responsible for getting devlink info message
// this is introduced for test purpose
type devlinkInfoGetter func(bus, device string) ([]byte, error)

// DevlinkGetDeviceInfoByName returns devlink info for selected device,
// otherwise returns an error code.
// Equivalent to: `devlink dev info $dev`
func (h *Handle) DevlinkGetDeviceInfoByName(Bus string, Device string, getInfoMsg devlinkInfoGetter) (*DevlinkDeviceInfo, error) {
	info, err := h.DevlinkGetDeviceInfoByNameAsMap(Bus, Device, getInfoMsg)
	if err != nil {
		return nil, err
	}

	return parseInfoData(info), nil
}

// DevlinkGetDeviceInfoByName returns devlink info for selected device,
// otherwise returns an error code.
// Equivalent to: `devlink dev info $dev`
func DevlinkGetDeviceInfoByName(Bus string, Device string) (*DevlinkDeviceInfo, error) {
	return pkgHandle.DevlinkGetDeviceInfoByName(Bus, Device, pkgHandle.getDevlinkInfoMsg)
}

// DevlinkGetDeviceInfoByNameAsMap returns devlink info for selected device as a map,
// otherwise returns an error code.
// Equivalent to: `devlink dev info $dev`
func (h *Handle) DevlinkGetDeviceInfoByNameAsMap(Bus string, Device string, getInfoMsg devlinkInfoGetter) (map[string]string, error) {
	response, err := getInfoMsg(Bus, Device)
	if err != nil {
		return nil, err
	}

	info, err := parseInfoMsg(response)
	if err != nil {
		return nil, err
	}

	return info, nil
}

// DevlinkGetDeviceInfoByNameAsMap returns devlink info for selected device as a map,
// otherwise returns an error code.
// Equivalent to: `devlink dev info $dev`
func DevlinkGetDeviceInfoByNameAsMap(Bus string, Device string) (map[string]string, error) {
	return pkgHandle.DevlinkGetDeviceInfoByNameAsMap(Bus, Device, pkgHandle.getDevlinkInfoMsg)
}

// GetDevlinkInfo returns devlink info for target device,
// otherwise returns an error code.
func (d *DevlinkDevice) GetDevlinkInfo() (*DevlinkDeviceInfo, error) {
	return pkgHandle.DevlinkGetDeviceInfoByName(d.BusName, d.DeviceName, pkgHandle.getDevlinkInfoMsg)
}

// GetDevlinkInfoAsMap returns devlink info for target device as a map,
// otherwise returns an error code.
func (d *DevlinkDevice) GetDevlinkInfoAsMap() (map[string]string, error) {
	return pkgHandle.DevlinkGetDeviceInfoByNameAsMap(d.BusName, d.DeviceName, pkgHandle.getDevlinkInfoMsg)
}

func (h *Handle) getDevlinkInfoMsg(bus, device string) ([]byte, error) {
	_, req, err := h.createCmdReq(nl.DEVLINK_CMD_INFO_GET, bus, device)
	if err != nil {
		return nil, err
	}

	response, err := req.Execute(unix.NETLINK_GENERIC, 0)
	if err != nil {
		return nil, err
	}

	if len(response) < 1 {
		return nil, fmt.Errorf("getDevlinkInfoMsg: message too short")
	}

	return response[0], nil
}

func parseInfoMsg(msg []byte) (map[string]string, error) {
	if len(msg) < nl.SizeofGenlmsg {
		return nil, fmt.Errorf("parseInfoMsg: message too short")
	}

	info := make(map[string]string)
	err := collectInfoData(msg[nl.SizeofGenlmsg:], info)

	if err != nil {
		return nil, err
	}

	return info, nil
}

func collectInfoData(msg []byte, data map[string]string) error {
	attrs, err := nl.ParseRouteAttr(msg)
	if err != nil {
		return err
	}

	for _, attr := range attrs {
		switch attr.Attr.Type {
		case nl.DEVLINK_ATTR_INFO_DRIVER_NAME:
			data["driver"] = parseInfoValue(attr.Value)
		case nl.DEVLINK_ATTR_INFO_SERIAL_NUMBER:
			data["serialNumber"] = parseInfoValue(attr.Value)
		case nl.DEVLINK_ATTR_INFO_VERSION_RUNNING, nl.DEVLINK_ATTR_INFO_VERSION_FIXED,
			nl.DEVLINK_ATTR_INFO_VERSION_STORED:
			key, value, err := getNestedInfoData(attr.Value)
			if err != nil {
				return err
			}
			data[key] = value
		}
	}

	if len(data) == 0 {
		return fmt.Errorf("collectInfoData: could not read attributes")
	}

	return nil
}

func getNestedInfoData(msg []byte) (string, string, error) {
	nestedAttrs, err := nl.ParseRouteAttr(msg)

	var key, value string

	if err != nil {
		return "", "", err
	}

	if len(nestedAttrs) != 2 {
		return "", "", fmt.Errorf("getNestedInfoData: too few attributes in nested structure")
	}

	for _, nestedAttr := range nestedAttrs {
		switch nestedAttr.Attr.Type {
		case nl.DEVLINK_ATTR_INFO_VERSION_NAME:
			key = parseInfoValue(nestedAttr.Value)
		case nl.DEVLINK_ATTR_INFO_VERSION_VALUE:
			value = parseInfoValue(nestedAttr.Value)
		}
	}

	if key == "" {
		return "", "", fmt.Errorf("getNestedInfoData: key not found")
	}

	if value == "" {
		return "", "", fmt.Errorf("getNestedInfoData: value not found")
	}

	return key, value, nil
}

func parseInfoData(data map[string]string) *DevlinkDeviceInfo {
	info := new(DevlinkDeviceInfo)
	for key, value := range data {
		switch key {
		case "driver":
			info.Driver = value
		case "serialNumber":
			info.SerialNumber = value
		case "board.id":
			info.BoardID = value
		case "fw.app":
			info.FwApp = value
		case "fw.app.bundle_id":
			info.FwAppBoundleID = value
		case "fw.app.name":
			info.FwAppName = value
		case "fw.bundle_id":
			info.FwBoundleID = value
		case "fw.mgmt":
			info.FwMgmt = value
		case "fw.mgmt.api":
			info.FwMgmtAPI = value
		case "fw.mgmt.build":
			info.FwMgmtBuild = value
		case "fw.netlist":
			info.FwNetlist = value
		case "fw.netlist.build":
			info.FwNetlistBuild = value
		case "fw.psid.api":
			info.FwPsidAPI = value
		case "fw.undi":
			info.FwUndi = value
		}
	}
	return info
}

func parseInfoValue(value []byte) string {
	v := strings.ReplaceAll(string(value), "\x00", "")
	return strings.TrimSpace(v)
}

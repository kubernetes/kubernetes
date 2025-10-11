// Copyright 2019 (c) Microsoft Corporation.
// Licensed under the MIT license.

//
// Author:
//      Auto Generated on 9/18/2020 using wmigen
//      Source root.CIMV2
//////////////////////////////////////////////
package cimv2

import (
	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/pkg/errors"
	cim "github.com/microsoft/wmi/pkg/wmiinstance"
	"reflect"
)

// Win32_NetworkAdapterConfiguration struct
type Win32_NetworkAdapterConfiguration struct {
	*CIM_Setting

	//
	ArpAlwaysSourceRoute bool

	//
	ArpUseEtherSNAP bool

	//
	DatabasePath string

	//
	DeadGWDetectEnabled bool

	//
	DefaultIPGateway []string

	//
	DefaultTOS uint8

	//
	DefaultTTL uint8

	//
	DHCPEnabled bool

	//
	DHCPLeaseExpires string

	//
	DHCPLeaseObtained string

	//
	DHCPServer string

	//
	DNSDomain string

	//
	DNSDomainSuffixSearchOrder []string

	//
	DNSEnabledForWINSResolution bool

	//
	DNSHostName string

	//
	DNSServerSearchOrder []string

	//
	DomainDNSRegistrationEnabled bool

	//
	ForwardBufferMemory uint32

	//
	FullDNSRegistrationEnabled bool

	//
	GatewayCostMetric []uint16

	//
	IGMPLevel uint8

	//
	Index uint32

	//
	InterfaceIndex uint32

	//
	IPAddress []string

	//
	IPConnectionMetric uint32

	//
	IPEnabled bool

	//
	IPFilterSecurityEnabled bool

	//
	IPPortSecurityEnabled bool

	//
	IPSecPermitIPProtocols []string

	//
	IPSecPermitTCPPorts []string

	//
	IPSecPermitUDPPorts []string

	//
	IPSubnet []string

	//
	IPUseZeroBroadcast bool

	//
	IPXAddress string

	//
	IPXEnabled bool

	//
	IPXFrameType []uint32

	//
	IPXMediaType uint32

	//
	IPXNetworkNumber []string

	//
	IPXVirtualNetNumber string

	//
	KeepAliveInterval uint32

	//
	KeepAliveTime uint32

	//
	MACAddress string

	//
	MTU uint32

	//
	NumForwardPackets uint32

	//
	PMTUBHDetectEnabled bool

	//
	PMTUDiscoveryEnabled bool

	//
	ServiceName string

	//
	TcpipNetbiosOptions uint32

	//
	TcpMaxConnectRetransmissions uint32

	//
	TcpMaxDataRetransmissions uint32

	//
	TcpNumConnections uint32

	//
	TcpUseRFC1122UrgentPointer bool

	//
	TcpWindowSize uint16

	//
	WINSEnableLMHostsLookup bool

	//
	WINSHostLookupFile string

	//
	WINSPrimaryServer string

	//
	WINSScopeID string

	//
	WINSSecondaryServer string
}

func NewWin32_NetworkAdapterConfigurationEx1(instance *cim.WmiInstance) (newInstance *Win32_NetworkAdapterConfiguration, err error) {
	tmp, err := NewCIM_SettingEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_NetworkAdapterConfiguration{
		CIM_Setting: tmp,
	}
	return
}

func NewWin32_NetworkAdapterConfigurationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_NetworkAdapterConfiguration, err error) {
	tmp, err := NewCIM_SettingEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_NetworkAdapterConfiguration{
		CIM_Setting: tmp,
	}
	return
}

// SetArpAlwaysSourceRoute sets the value of ArpAlwaysSourceRoute for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyArpAlwaysSourceRoute(value bool) (err error) {
	return instance.SetProperty("ArpAlwaysSourceRoute", (value))
}

// GetArpAlwaysSourceRoute gets the value of ArpAlwaysSourceRoute for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyArpAlwaysSourceRoute() (value bool, err error) {
	retValue, err := instance.GetProperty("ArpAlwaysSourceRoute")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetArpUseEtherSNAP sets the value of ArpUseEtherSNAP for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyArpUseEtherSNAP(value bool) (err error) {
	return instance.SetProperty("ArpUseEtherSNAP", (value))
}

// GetArpUseEtherSNAP gets the value of ArpUseEtherSNAP for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyArpUseEtherSNAP() (value bool, err error) {
	retValue, err := instance.GetProperty("ArpUseEtherSNAP")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetDatabasePath sets the value of DatabasePath for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDatabasePath(value string) (err error) {
	return instance.SetProperty("DatabasePath", (value))
}

// GetDatabasePath gets the value of DatabasePath for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDatabasePath() (value string, err error) {
	retValue, err := instance.GetProperty("DatabasePath")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetDeadGWDetectEnabled sets the value of DeadGWDetectEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDeadGWDetectEnabled(value bool) (err error) {
	return instance.SetProperty("DeadGWDetectEnabled", (value))
}

// GetDeadGWDetectEnabled gets the value of DeadGWDetectEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDeadGWDetectEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("DeadGWDetectEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetDefaultIPGateway sets the value of DefaultIPGateway for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDefaultIPGateway(value []string) (err error) {
	return instance.SetProperty("DefaultIPGateway", (value))
}

// GetDefaultIPGateway gets the value of DefaultIPGateway for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDefaultIPGateway() (value []string, err error) {
	retValue, err := instance.GetProperty("DefaultIPGateway")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetDefaultTOS sets the value of DefaultTOS for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDefaultTOS(value uint8) (err error) {
	return instance.SetProperty("DefaultTOS", (value))
}

// GetDefaultTOS gets the value of DefaultTOS for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDefaultTOS() (value uint8, err error) {
	retValue, err := instance.GetProperty("DefaultTOS")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetDefaultTTL sets the value of DefaultTTL for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDefaultTTL(value uint8) (err error) {
	return instance.SetProperty("DefaultTTL", (value))
}

// GetDefaultTTL gets the value of DefaultTTL for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDefaultTTL() (value uint8, err error) {
	retValue, err := instance.GetProperty("DefaultTTL")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetDHCPEnabled sets the value of DHCPEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDHCPEnabled(value bool) (err error) {
	return instance.SetProperty("DHCPEnabled", (value))
}

// GetDHCPEnabled gets the value of DHCPEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDHCPEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("DHCPEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetDHCPLeaseExpires sets the value of DHCPLeaseExpires for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDHCPLeaseExpires(value string) (err error) {
	return instance.SetProperty("DHCPLeaseExpires", (value))
}

// GetDHCPLeaseExpires gets the value of DHCPLeaseExpires for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDHCPLeaseExpires() (value string, err error) {
	retValue, err := instance.GetProperty("DHCPLeaseExpires")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetDHCPLeaseObtained sets the value of DHCPLeaseObtained for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDHCPLeaseObtained(value string) (err error) {
	return instance.SetProperty("DHCPLeaseObtained", (value))
}

// GetDHCPLeaseObtained gets the value of DHCPLeaseObtained for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDHCPLeaseObtained() (value string, err error) {
	retValue, err := instance.GetProperty("DHCPLeaseObtained")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetDHCPServer sets the value of DHCPServer for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDHCPServer(value string) (err error) {
	return instance.SetProperty("DHCPServer", (value))
}

// GetDHCPServer gets the value of DHCPServer for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDHCPServer() (value string, err error) {
	retValue, err := instance.GetProperty("DHCPServer")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetDNSDomain sets the value of DNSDomain for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDNSDomain(value string) (err error) {
	return instance.SetProperty("DNSDomain", (value))
}

// GetDNSDomain gets the value of DNSDomain for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDNSDomain() (value string, err error) {
	retValue, err := instance.GetProperty("DNSDomain")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetDNSDomainSuffixSearchOrder sets the value of DNSDomainSuffixSearchOrder for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDNSDomainSuffixSearchOrder(value []string) (err error) {
	return instance.SetProperty("DNSDomainSuffixSearchOrder", (value))
}

// GetDNSDomainSuffixSearchOrder gets the value of DNSDomainSuffixSearchOrder for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDNSDomainSuffixSearchOrder() (value []string, err error) {
	retValue, err := instance.GetProperty("DNSDomainSuffixSearchOrder")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetDNSEnabledForWINSResolution sets the value of DNSEnabledForWINSResolution for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDNSEnabledForWINSResolution(value bool) (err error) {
	return instance.SetProperty("DNSEnabledForWINSResolution", (value))
}

// GetDNSEnabledForWINSResolution gets the value of DNSEnabledForWINSResolution for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDNSEnabledForWINSResolution() (value bool, err error) {
	retValue, err := instance.GetProperty("DNSEnabledForWINSResolution")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetDNSHostName sets the value of DNSHostName for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDNSHostName(value string) (err error) {
	return instance.SetProperty("DNSHostName", (value))
}

// GetDNSHostName gets the value of DNSHostName for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDNSHostName() (value string, err error) {
	retValue, err := instance.GetProperty("DNSHostName")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetDNSServerSearchOrder sets the value of DNSServerSearchOrder for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDNSServerSearchOrder(value []string) (err error) {
	return instance.SetProperty("DNSServerSearchOrder", (value))
}

// GetDNSServerSearchOrder gets the value of DNSServerSearchOrder for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDNSServerSearchOrder() (value []string, err error) {
	retValue, err := instance.GetProperty("DNSServerSearchOrder")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetDomainDNSRegistrationEnabled sets the value of DomainDNSRegistrationEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyDomainDNSRegistrationEnabled(value bool) (err error) {
	return instance.SetProperty("DomainDNSRegistrationEnabled", (value))
}

// GetDomainDNSRegistrationEnabled gets the value of DomainDNSRegistrationEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyDomainDNSRegistrationEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("DomainDNSRegistrationEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetForwardBufferMemory sets the value of ForwardBufferMemory for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyForwardBufferMemory(value uint32) (err error) {
	return instance.SetProperty("ForwardBufferMemory", (value))
}

// GetForwardBufferMemory gets the value of ForwardBufferMemory for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyForwardBufferMemory() (value uint32, err error) {
	retValue, err := instance.GetProperty("ForwardBufferMemory")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetFullDNSRegistrationEnabled sets the value of FullDNSRegistrationEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyFullDNSRegistrationEnabled(value bool) (err error) {
	return instance.SetProperty("FullDNSRegistrationEnabled", (value))
}

// GetFullDNSRegistrationEnabled gets the value of FullDNSRegistrationEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyFullDNSRegistrationEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("FullDNSRegistrationEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetGatewayCostMetric sets the value of GatewayCostMetric for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyGatewayCostMetric(value []uint16) (err error) {
	return instance.SetProperty("GatewayCostMetric", (value))
}

// GetGatewayCostMetric gets the value of GatewayCostMetric for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyGatewayCostMetric() (value []uint16, err error) {
	retValue, err := instance.GetProperty("GatewayCostMetric")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint16)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint16(valuetmp))
	}

	return
}

// SetIGMPLevel sets the value of IGMPLevel for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIGMPLevel(value uint8) (err error) {
	return instance.SetProperty("IGMPLevel", (value))
}

// GetIGMPLevel gets the value of IGMPLevel for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIGMPLevel() (value uint8, err error) {
	retValue, err := instance.GetProperty("IGMPLevel")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint8)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint8 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint8(valuetmp)

	return
}

// SetIndex sets the value of Index for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIndex(value uint32) (err error) {
	return instance.SetProperty("Index", (value))
}

// GetIndex gets the value of Index for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIndex() (value uint32, err error) {
	retValue, err := instance.GetProperty("Index")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetInterfaceIndex sets the value of InterfaceIndex for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyInterfaceIndex(value uint32) (err error) {
	return instance.SetProperty("InterfaceIndex", (value))
}

// GetInterfaceIndex gets the value of InterfaceIndex for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyInterfaceIndex() (value uint32, err error) {
	retValue, err := instance.GetProperty("InterfaceIndex")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetIPAddress sets the value of IPAddress for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPAddress(value []string) (err error) {
	return instance.SetProperty("IPAddress", (value))
}

// GetIPAddress gets the value of IPAddress for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPAddress() (value []string, err error) {
	retValue, err := instance.GetProperty("IPAddress")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetIPConnectionMetric sets the value of IPConnectionMetric for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPConnectionMetric(value uint32) (err error) {
	return instance.SetProperty("IPConnectionMetric", (value))
}

// GetIPConnectionMetric gets the value of IPConnectionMetric for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPConnectionMetric() (value uint32, err error) {
	retValue, err := instance.GetProperty("IPConnectionMetric")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetIPEnabled sets the value of IPEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPEnabled(value bool) (err error) {
	return instance.SetProperty("IPEnabled", (value))
}

// GetIPEnabled gets the value of IPEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("IPEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIPFilterSecurityEnabled sets the value of IPFilterSecurityEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPFilterSecurityEnabled(value bool) (err error) {
	return instance.SetProperty("IPFilterSecurityEnabled", (value))
}

// GetIPFilterSecurityEnabled gets the value of IPFilterSecurityEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPFilterSecurityEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("IPFilterSecurityEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIPPortSecurityEnabled sets the value of IPPortSecurityEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPPortSecurityEnabled(value bool) (err error) {
	return instance.SetProperty("IPPortSecurityEnabled", (value))
}

// GetIPPortSecurityEnabled gets the value of IPPortSecurityEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPPortSecurityEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("IPPortSecurityEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIPSecPermitIPProtocols sets the value of IPSecPermitIPProtocols for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPSecPermitIPProtocols(value []string) (err error) {
	return instance.SetProperty("IPSecPermitIPProtocols", (value))
}

// GetIPSecPermitIPProtocols gets the value of IPSecPermitIPProtocols for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPSecPermitIPProtocols() (value []string, err error) {
	retValue, err := instance.GetProperty("IPSecPermitIPProtocols")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetIPSecPermitTCPPorts sets the value of IPSecPermitTCPPorts for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPSecPermitTCPPorts(value []string) (err error) {
	return instance.SetProperty("IPSecPermitTCPPorts", (value))
}

// GetIPSecPermitTCPPorts gets the value of IPSecPermitTCPPorts for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPSecPermitTCPPorts() (value []string, err error) {
	retValue, err := instance.GetProperty("IPSecPermitTCPPorts")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetIPSecPermitUDPPorts sets the value of IPSecPermitUDPPorts for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPSecPermitUDPPorts(value []string) (err error) {
	return instance.SetProperty("IPSecPermitUDPPorts", (value))
}

// GetIPSecPermitUDPPorts gets the value of IPSecPermitUDPPorts for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPSecPermitUDPPorts() (value []string, err error) {
	retValue, err := instance.GetProperty("IPSecPermitUDPPorts")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetIPSubnet sets the value of IPSubnet for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPSubnet(value []string) (err error) {
	return instance.SetProperty("IPSubnet", (value))
}

// GetIPSubnet gets the value of IPSubnet for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPSubnet() (value []string, err error) {
	retValue, err := instance.GetProperty("IPSubnet")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetIPUseZeroBroadcast sets the value of IPUseZeroBroadcast for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPUseZeroBroadcast(value bool) (err error) {
	return instance.SetProperty("IPUseZeroBroadcast", (value))
}

// GetIPUseZeroBroadcast gets the value of IPUseZeroBroadcast for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPUseZeroBroadcast() (value bool, err error) {
	retValue, err := instance.GetProperty("IPUseZeroBroadcast")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIPXAddress sets the value of IPXAddress for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPXAddress(value string) (err error) {
	return instance.SetProperty("IPXAddress", (value))
}

// GetIPXAddress gets the value of IPXAddress for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPXAddress() (value string, err error) {
	retValue, err := instance.GetProperty("IPXAddress")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetIPXEnabled sets the value of IPXEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPXEnabled(value bool) (err error) {
	return instance.SetProperty("IPXEnabled", (value))
}

// GetIPXEnabled gets the value of IPXEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPXEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("IPXEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetIPXFrameType sets the value of IPXFrameType for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPXFrameType(value []uint32) (err error) {
	return instance.SetProperty("IPXFrameType", (value))
}

// GetIPXFrameType gets the value of IPXFrameType for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPXFrameType() (value []uint32, err error) {
	retValue, err := instance.GetProperty("IPXFrameType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(uint32)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, uint32(valuetmp))
	}

	return
}

// SetIPXMediaType sets the value of IPXMediaType for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPXMediaType(value uint32) (err error) {
	return instance.SetProperty("IPXMediaType", (value))
}

// GetIPXMediaType gets the value of IPXMediaType for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPXMediaType() (value uint32, err error) {
	retValue, err := instance.GetProperty("IPXMediaType")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetIPXNetworkNumber sets the value of IPXNetworkNumber for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPXNetworkNumber(value []string) (err error) {
	return instance.SetProperty("IPXNetworkNumber", (value))
}

// GetIPXNetworkNumber gets the value of IPXNetworkNumber for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPXNetworkNumber() (value []string, err error) {
	retValue, err := instance.GetProperty("IPXNetworkNumber")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	for _, interfaceValue := range retValue.([]interface{}) {
		valuetmp, ok := interfaceValue.(string)
		if !ok {
			err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(interfaceValue))
			return
		}
		value = append(value, string(valuetmp))
	}

	return
}

// SetIPXVirtualNetNumber sets the value of IPXVirtualNetNumber for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyIPXVirtualNetNumber(value string) (err error) {
	return instance.SetProperty("IPXVirtualNetNumber", (value))
}

// GetIPXVirtualNetNumber gets the value of IPXVirtualNetNumber for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyIPXVirtualNetNumber() (value string, err error) {
	retValue, err := instance.GetProperty("IPXVirtualNetNumber")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetKeepAliveInterval sets the value of KeepAliveInterval for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyKeepAliveInterval(value uint32) (err error) {
	return instance.SetProperty("KeepAliveInterval", (value))
}

// GetKeepAliveInterval gets the value of KeepAliveInterval for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyKeepAliveInterval() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeepAliveInterval")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetKeepAliveTime sets the value of KeepAliveTime for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyKeepAliveTime(value uint32) (err error) {
	return instance.SetProperty("KeepAliveTime", (value))
}

// GetKeepAliveTime gets the value of KeepAliveTime for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyKeepAliveTime() (value uint32, err error) {
	retValue, err := instance.GetProperty("KeepAliveTime")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetMACAddress sets the value of MACAddress for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyMACAddress(value string) (err error) {
	return instance.SetProperty("MACAddress", (value))
}

// GetMACAddress gets the value of MACAddress for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyMACAddress() (value string, err error) {
	retValue, err := instance.GetProperty("MACAddress")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetMTU sets the value of MTU for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyMTU(value uint32) (err error) {
	return instance.SetProperty("MTU", (value))
}

// GetMTU gets the value of MTU for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyMTU() (value uint32, err error) {
	retValue, err := instance.GetProperty("MTU")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetNumForwardPackets sets the value of NumForwardPackets for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyNumForwardPackets(value uint32) (err error) {
	return instance.SetProperty("NumForwardPackets", (value))
}

// GetNumForwardPackets gets the value of NumForwardPackets for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyNumForwardPackets() (value uint32, err error) {
	retValue, err := instance.GetProperty("NumForwardPackets")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetPMTUBHDetectEnabled sets the value of PMTUBHDetectEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyPMTUBHDetectEnabled(value bool) (err error) {
	return instance.SetProperty("PMTUBHDetectEnabled", (value))
}

// GetPMTUBHDetectEnabled gets the value of PMTUBHDetectEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyPMTUBHDetectEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("PMTUBHDetectEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetPMTUDiscoveryEnabled sets the value of PMTUDiscoveryEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyPMTUDiscoveryEnabled(value bool) (err error) {
	return instance.SetProperty("PMTUDiscoveryEnabled", (value))
}

// GetPMTUDiscoveryEnabled gets the value of PMTUDiscoveryEnabled for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyPMTUDiscoveryEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("PMTUDiscoveryEnabled")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetServiceName sets the value of ServiceName for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyServiceName(value string) (err error) {
	return instance.SetProperty("ServiceName", (value))
}

// GetServiceName gets the value of ServiceName for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyServiceName() (value string, err error) {
	retValue, err := instance.GetProperty("ServiceName")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetTcpipNetbiosOptions sets the value of TcpipNetbiosOptions for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyTcpipNetbiosOptions(value uint32) (err error) {
	return instance.SetProperty("TcpipNetbiosOptions", (value))
}

// GetTcpipNetbiosOptions gets the value of TcpipNetbiosOptions for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyTcpipNetbiosOptions() (value uint32, err error) {
	retValue, err := instance.GetProperty("TcpipNetbiosOptions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetTcpMaxConnectRetransmissions sets the value of TcpMaxConnectRetransmissions for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyTcpMaxConnectRetransmissions(value uint32) (err error) {
	return instance.SetProperty("TcpMaxConnectRetransmissions", (value))
}

// GetTcpMaxConnectRetransmissions gets the value of TcpMaxConnectRetransmissions for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyTcpMaxConnectRetransmissions() (value uint32, err error) {
	retValue, err := instance.GetProperty("TcpMaxConnectRetransmissions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetTcpMaxDataRetransmissions sets the value of TcpMaxDataRetransmissions for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyTcpMaxDataRetransmissions(value uint32) (err error) {
	return instance.SetProperty("TcpMaxDataRetransmissions", (value))
}

// GetTcpMaxDataRetransmissions gets the value of TcpMaxDataRetransmissions for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyTcpMaxDataRetransmissions() (value uint32, err error) {
	retValue, err := instance.GetProperty("TcpMaxDataRetransmissions")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetTcpNumConnections sets the value of TcpNumConnections for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyTcpNumConnections(value uint32) (err error) {
	return instance.SetProperty("TcpNumConnections", (value))
}

// GetTcpNumConnections gets the value of TcpNumConnections for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyTcpNumConnections() (value uint32, err error) {
	retValue, err := instance.GetProperty("TcpNumConnections")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint32(valuetmp)

	return
}

// SetTcpUseRFC1122UrgentPointer sets the value of TcpUseRFC1122UrgentPointer for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyTcpUseRFC1122UrgentPointer(value bool) (err error) {
	return instance.SetProperty("TcpUseRFC1122UrgentPointer", (value))
}

// GetTcpUseRFC1122UrgentPointer gets the value of TcpUseRFC1122UrgentPointer for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyTcpUseRFC1122UrgentPointer() (value bool, err error) {
	retValue, err := instance.GetProperty("TcpUseRFC1122UrgentPointer")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetTcpWindowSize sets the value of TcpWindowSize for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyTcpWindowSize(value uint16) (err error) {
	return instance.SetProperty("TcpWindowSize", (value))
}

// GetTcpWindowSize gets the value of TcpWindowSize for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyTcpWindowSize() (value uint16, err error) {
	retValue, err := instance.GetProperty("TcpWindowSize")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint16)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint16 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint16(valuetmp)

	return
}

// SetWINSEnableLMHostsLookup sets the value of WINSEnableLMHostsLookup for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyWINSEnableLMHostsLookup(value bool) (err error) {
	return instance.SetProperty("WINSEnableLMHostsLookup", (value))
}

// GetWINSEnableLMHostsLookup gets the value of WINSEnableLMHostsLookup for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyWINSEnableLMHostsLookup() (value bool, err error) {
	retValue, err := instance.GetProperty("WINSEnableLMHostsLookup")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(bool)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " bool is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = bool(valuetmp)

	return
}

// SetWINSHostLookupFile sets the value of WINSHostLookupFile for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyWINSHostLookupFile(value string) (err error) {
	return instance.SetProperty("WINSHostLookupFile", (value))
}

// GetWINSHostLookupFile gets the value of WINSHostLookupFile for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyWINSHostLookupFile() (value string, err error) {
	retValue, err := instance.GetProperty("WINSHostLookupFile")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetWINSPrimaryServer sets the value of WINSPrimaryServer for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyWINSPrimaryServer(value string) (err error) {
	return instance.SetProperty("WINSPrimaryServer", (value))
}

// GetWINSPrimaryServer gets the value of WINSPrimaryServer for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyWINSPrimaryServer() (value string, err error) {
	retValue, err := instance.GetProperty("WINSPrimaryServer")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetWINSScopeID sets the value of WINSScopeID for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyWINSScopeID(value string) (err error) {
	return instance.SetProperty("WINSScopeID", (value))
}

// GetWINSScopeID gets the value of WINSScopeID for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyWINSScopeID() (value string, err error) {
	retValue, err := instance.GetProperty("WINSScopeID")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

// SetWINSSecondaryServer sets the value of WINSSecondaryServer for the instance
func (instance *Win32_NetworkAdapterConfiguration) SetPropertyWINSSecondaryServer(value string) (err error) {
	return instance.SetProperty("WINSSecondaryServer", (value))
}

// GetWINSSecondaryServer gets the value of WINSSecondaryServer for the instance
func (instance *Win32_NetworkAdapterConfiguration) GetPropertyWINSSecondaryServer() (value string, err error) {
	retValue, err := instance.GetProperty("WINSSecondaryServer")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(string)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " string is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = string(valuetmp)

	return
}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) EnableDHCP() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("EnableDHCP")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) RenewDHCPLease() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("RenewDHCPLease")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) RenewDHCPLeaseAll() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("RenewDHCPLeaseAll")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) ReleaseDHCPLease() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ReleaseDHCPLease")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) ReleaseDHCPLeaseAll() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("ReleaseDHCPLeaseAll")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="IPAddress" type="string []"></param>
// <param name="SubnetMask" type="string []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) EnableStatic( /* IN */ IPAddress []string,
	/* IN */ SubnetMask []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("EnableStatic", IPAddress, SubnetMask)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DefaultIPGateway" type="string []"></param>
// <param name="GatewayCostMetric" type="uint16 []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetGateways( /* IN */ DefaultIPGateway []string,
	/* OPTIONAL IN */ GatewayCostMetric []uint16) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetGateways", DefaultIPGateway, GatewayCostMetric)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DNSDomain" type="string "></param>
// <param name="DNSDomainSuffixSearchOrder" type="string []"></param>
// <param name="DNSHostName" type="string "></param>
// <param name="DNSServerSearchOrder" type="string []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) EnableDNS( /* OPTIONAL IN */ DNSHostName string,
	/* OPTIONAL IN */ DNSDomain string,
	/* OPTIONAL IN */ DNSServerSearchOrder []string,
	/* OPTIONAL IN */ DNSDomainSuffixSearchOrder []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("EnableDNS", DNSHostName, DNSDomain, DNSServerSearchOrder, DNSDomainSuffixSearchOrder)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DNSDomain" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetDNSDomain( /* IN */ DNSDomain string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDNSDomain", DNSDomain)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DNSServerSearchOrder" type="string []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetDNSServerSearchOrder( /* IN */ DNSServerSearchOrder []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDNSServerSearchOrder", DNSServerSearchOrder)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DNSDomainSuffixSearchOrder" type="string []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetDNSSuffixSearchOrder( /* IN */ DNSDomainSuffixSearchOrder []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDNSSuffixSearchOrder", DNSDomainSuffixSearchOrder)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DomainDNSRegistrationEnabled" type="bool "></param>
// <param name="FullDNSRegistrationEnabled" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetDynamicDNSRegistration( /* IN */ FullDNSRegistrationEnabled bool,
	/* OPTIONAL IN */ DomainDNSRegistrationEnabled bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDynamicDNSRegistration", FullDNSRegistrationEnabled, DomainDNSRegistrationEnabled)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="IPConnectionMetric" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetIPConnectionMetric( /* IN */ IPConnectionMetric uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetIPConnectionMetric", IPConnectionMetric)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="WINSPrimaryServer" type="string "></param>
// <param name="WINSSecondaryServer" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetWINSServer( /* IN */ WINSPrimaryServer string,
	/* IN */ WINSSecondaryServer string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetWINSServer", WINSPrimaryServer, WINSSecondaryServer)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DNSEnabledForWINSResolution" type="bool "></param>
// <param name="WINSEnableLMHostsLookup" type="bool "></param>
// <param name="WINSHostLookupFile" type="string "></param>
// <param name="WINSScopeID" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) EnableWINS( /* IN */ DNSEnabledForWINSResolution bool,
	/* IN */ WINSEnableLMHostsLookup bool,
	/* OPTIONAL IN */ WINSHostLookupFile string,
	/* OPTIONAL IN */ WINSScopeID string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("EnableWINS", DNSEnabledForWINSResolution, WINSEnableLMHostsLookup, WINSHostLookupFile, WINSScopeID)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="TcpipNetbiosOptions" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetTcpipNetbios( /* IN */ TcpipNetbiosOptions uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetTcpipNetbios", TcpipNetbiosOptions)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="IPSecPermitIPProtocols" type="string []"></param>
// <param name="IPSecPermitTCPPorts" type="string []"></param>
// <param name="IPSecPermitUDPPorts" type="string []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) EnableIPSec( /* IN */ IPSecPermitTCPPorts []string,
	/* IN */ IPSecPermitUDPPorts []string,
	/* IN */ IPSecPermitIPProtocols []string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("EnableIPSec", IPSecPermitTCPPorts, IPSecPermitUDPPorts, IPSecPermitIPProtocols)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) DisableIPSec() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("DisableIPSec")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="IPXVirtualNetNumber" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetIPXVirtualNetworkNumber( /* IN */ IPXVirtualNetNumber string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetIPXVirtualNetworkNumber", IPXVirtualNetNumber)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="IPXFrameType" type="uint32 []"></param>
// <param name="IPXNetworkNumber" type="string []"></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetIPXFrameTypeNetworkPairs( /* IN */ IPXNetworkNumber []string,
	/* IN */ IPXFrameType []uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetIPXFrameTypeNetworkPairs", IPXNetworkNumber, IPXFrameType)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DatabasePath" type="string "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetDatabasePath( /* IN */ DatabasePath string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDatabasePath", DatabasePath)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="IPUseZeroBroadcast" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetIPUseZeroBroadcast( /* IN */ IPUseZeroBroadcast bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetIPUseZeroBroadcast", IPUseZeroBroadcast)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ArpAlwaysSourceRoute" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetArpAlwaysSourceRoute( /* IN */ ArpAlwaysSourceRoute bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetArpAlwaysSourceRoute", ArpAlwaysSourceRoute)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ArpUseEtherSNAP" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetArpUseEtherSNAP( /* IN */ ArpUseEtherSNAP bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetArpUseEtherSNAP", ArpUseEtherSNAP)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DefaultTOS" type="uint8 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetDefaultTOS( /* IN */ DefaultTOS uint8) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDefaultTOS", DefaultTOS)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DefaultTTL" type="uint8 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetDefaultTTL( /* IN */ DefaultTTL uint8) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDefaultTTL", DefaultTTL)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="DeadGWDetectEnabled" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetDeadGWDetect( /* IN */ DeadGWDetectEnabled bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDeadGWDetect", DeadGWDetectEnabled)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="PMTUBHDetectEnabled" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetPMTUBHDetect( /* IN */ PMTUBHDetectEnabled bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetPMTUBHDetect", PMTUBHDetectEnabled)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="PMTUDiscoveryEnabled" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetPMTUDiscovery( /* IN */ PMTUDiscoveryEnabled bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetPMTUDiscovery", PMTUDiscoveryEnabled)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="ForwardBufferMemory" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetForwardBufferMemory( /* IN */ ForwardBufferMemory uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetForwardBufferMemory", ForwardBufferMemory)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="IGMPLevel" type="uint8 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetIGMPLevel( /* IN */ IGMPLevel uint8) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetIGMPLevel", IGMPLevel)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="KeepAliveInterval" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetKeepAliveInterval( /* IN */ KeepAliveInterval uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetKeepAliveInterval", KeepAliveInterval)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="KeepAliveTime" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetKeepAliveTime( /* IN */ KeepAliveTime uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetKeepAliveTime", KeepAliveTime)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="MTU" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetMTU( /* IN */ MTU uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetMTU", MTU)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="NumForwardPackets" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetNumForwardPackets( /* IN */ NumForwardPackets uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetNumForwardPackets", NumForwardPackets)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="TcpMaxConnectRetransmissions" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetTcpMaxConnectRetransmissions( /* IN */ TcpMaxConnectRetransmissions uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetTcpMaxConnectRetransmissions", TcpMaxConnectRetransmissions)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="TcpMaxDataRetransmissions" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetTcpMaxDataRetransmissions( /* IN */ TcpMaxDataRetransmissions uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetTcpMaxDataRetransmissions", TcpMaxDataRetransmissions)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="TcpNumConnections" type="uint32 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetTcpNumConnections( /* IN */ TcpNumConnections uint32) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetTcpNumConnections", TcpNumConnections)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="TcpUseRFC1122UrgentPointer" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetTcpUseRFC1122UrgentPointer( /* IN */ TcpUseRFC1122UrgentPointer bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetTcpUseRFC1122UrgentPointer", TcpUseRFC1122UrgentPointer)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="TcpWindowSize" type="uint16 "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) SetTcpWindowSize( /* IN */ TcpWindowSize uint16) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetTcpWindowSize", TcpWindowSize)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//

// <param name="IPFilterSecurityEnabled" type="bool "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_NetworkAdapterConfiguration) EnableIPFilterSec( /* IN */ IPFilterSecurityEnabled bool) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("EnableIPFilterSec", IPFilterSecurityEnabled)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

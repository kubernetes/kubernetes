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

// Win32_PerfRawData_PerfNet_Redirector struct
type Win32_PerfRawData_PerfNet_Redirector struct {
	*Win32_PerfRawData

	//
	BytesReceivedPersec uint64

	//
	BytesTotalPersec uint64

	//
	BytesTransmittedPersec uint64

	//
	ConnectsCore uint32

	//
	ConnectsLanManager20 uint32

	//
	ConnectsLanManager21 uint32

	//
	ConnectsWindowsNT uint32

	//
	CurrentCommands uint32

	//
	FileDataOperationsPersec uint32

	//
	FileReadOperationsPersec uint32

	//
	FileWriteOperationsPersec uint32

	//
	NetworkErrorsPersec uint32

	//
	PacketsPersec uint64

	//
	PacketsReceivedPersec uint64

	//
	PacketsTransmittedPersec uint64

	//
	ReadBytesCachePersec uint64

	//
	ReadBytesNetworkPersec uint64

	//
	ReadBytesNonPagingPersec uint64

	//
	ReadBytesPagingPersec uint64

	//
	ReadOperationsRandomPersec uint32

	//
	ReadPacketsPersec uint32

	//
	ReadPacketsSmallPersec uint32

	//
	ReadsDeniedPersec uint32

	//
	ReadsLargePersec uint32

	//
	ServerDisconnects uint32

	//
	ServerReconnects uint32

	//
	ServerSessions uint32

	//
	ServerSessionsHung uint32

	//
	WriteBytesCachePersec uint64

	//
	WriteBytesNetworkPersec uint64

	//
	WriteBytesNonPagingPersec uint64

	//
	WriteBytesPagingPersec uint64

	//
	WriteOperationsRandomPersec uint32

	//
	WritePacketsPersec uint32

	//
	WritePacketsSmallPersec uint32

	//
	WritesDeniedPersec uint32

	//
	WritesLargePersec uint32
}

func NewWin32_PerfRawData_PerfNet_RedirectorEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_PerfNet_Redirector, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_PerfNet_Redirector{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_PerfNet_RedirectorEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_PerfNet_Redirector, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_PerfNet_Redirector{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetBytesReceivedPersec sets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesReceivedPersec", (value))
}

// GetBytesReceivedPersec gets the value of BytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyBytesReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesReceivedPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetBytesTotalPersec sets the value of BytesTotalPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyBytesTotalPersec(value uint64) (err error) {
	return instance.SetProperty("BytesTotalPersec", (value))
}

// GetBytesTotalPersec gets the value of BytesTotalPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyBytesTotalPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTotalPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetBytesTransmittedPersec sets the value of BytesTransmittedPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyBytesTransmittedPersec(value uint64) (err error) {
	return instance.SetProperty("BytesTransmittedPersec", (value))
}

// GetBytesTransmittedPersec gets the value of BytesTransmittedPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyBytesTransmittedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("BytesTransmittedPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetConnectsCore sets the value of ConnectsCore for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyConnectsCore(value uint32) (err error) {
	return instance.SetProperty("ConnectsCore", (value))
}

// GetConnectsCore gets the value of ConnectsCore for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyConnectsCore() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectsCore")
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

// SetConnectsLanManager20 sets the value of ConnectsLanManager20 for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyConnectsLanManager20(value uint32) (err error) {
	return instance.SetProperty("ConnectsLanManager20", (value))
}

// GetConnectsLanManager20 gets the value of ConnectsLanManager20 for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyConnectsLanManager20() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectsLanManager20")
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

// SetConnectsLanManager21 sets the value of ConnectsLanManager21 for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyConnectsLanManager21(value uint32) (err error) {
	return instance.SetProperty("ConnectsLanManager21", (value))
}

// GetConnectsLanManager21 gets the value of ConnectsLanManager21 for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyConnectsLanManager21() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectsLanManager21")
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

// SetConnectsWindowsNT sets the value of ConnectsWindowsNT for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyConnectsWindowsNT(value uint32) (err error) {
	return instance.SetProperty("ConnectsWindowsNT", (value))
}

// GetConnectsWindowsNT gets the value of ConnectsWindowsNT for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyConnectsWindowsNT() (value uint32, err error) {
	retValue, err := instance.GetProperty("ConnectsWindowsNT")
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

// SetCurrentCommands sets the value of CurrentCommands for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyCurrentCommands(value uint32) (err error) {
	return instance.SetProperty("CurrentCommands", (value))
}

// GetCurrentCommands gets the value of CurrentCommands for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyCurrentCommands() (value uint32, err error) {
	retValue, err := instance.GetProperty("CurrentCommands")
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

// SetFileDataOperationsPersec sets the value of FileDataOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyFileDataOperationsPersec(value uint32) (err error) {
	return instance.SetProperty("FileDataOperationsPersec", (value))
}

// GetFileDataOperationsPersec gets the value of FileDataOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyFileDataOperationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FileDataOperationsPersec")
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

// SetFileReadOperationsPersec sets the value of FileReadOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyFileReadOperationsPersec(value uint32) (err error) {
	return instance.SetProperty("FileReadOperationsPersec", (value))
}

// GetFileReadOperationsPersec gets the value of FileReadOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyFileReadOperationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FileReadOperationsPersec")
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

// SetFileWriteOperationsPersec sets the value of FileWriteOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyFileWriteOperationsPersec(value uint32) (err error) {
	return instance.SetProperty("FileWriteOperationsPersec", (value))
}

// GetFileWriteOperationsPersec gets the value of FileWriteOperationsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyFileWriteOperationsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("FileWriteOperationsPersec")
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

// SetNetworkErrorsPersec sets the value of NetworkErrorsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyNetworkErrorsPersec(value uint32) (err error) {
	return instance.SetProperty("NetworkErrorsPersec", (value))
}

// GetNetworkErrorsPersec gets the value of NetworkErrorsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyNetworkErrorsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NetworkErrorsPersec")
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

// SetPacketsPersec sets the value of PacketsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyPacketsPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsPersec", (value))
}

// GetPacketsPersec gets the value of PacketsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyPacketsPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsReceivedPersec sets the value of PacketsReceivedPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyPacketsReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsReceivedPersec", (value))
}

// GetPacketsReceivedPersec gets the value of PacketsReceivedPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyPacketsReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsReceivedPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetPacketsTransmittedPersec sets the value of PacketsTransmittedPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyPacketsTransmittedPersec(value uint64) (err error) {
	return instance.SetProperty("PacketsTransmittedPersec", (value))
}

// GetPacketsTransmittedPersec gets the value of PacketsTransmittedPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyPacketsTransmittedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("PacketsTransmittedPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetReadBytesCachePersec sets the value of ReadBytesCachePersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyReadBytesCachePersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesCachePersec", (value))
}

// GetReadBytesCachePersec gets the value of ReadBytesCachePersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyReadBytesCachePersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBytesCachePersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetReadBytesNetworkPersec sets the value of ReadBytesNetworkPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyReadBytesNetworkPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesNetworkPersec", (value))
}

// GetReadBytesNetworkPersec gets the value of ReadBytesNetworkPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyReadBytesNetworkPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBytesNetworkPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetReadBytesNonPagingPersec sets the value of ReadBytesNonPagingPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyReadBytesNonPagingPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesNonPagingPersec", (value))
}

// GetReadBytesNonPagingPersec gets the value of ReadBytesNonPagingPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyReadBytesNonPagingPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBytesNonPagingPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetReadBytesPagingPersec sets the value of ReadBytesPagingPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyReadBytesPagingPersec(value uint64) (err error) {
	return instance.SetProperty("ReadBytesPagingPersec", (value))
}

// GetReadBytesPagingPersec gets the value of ReadBytesPagingPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyReadBytesPagingPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReadBytesPagingPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetReadOperationsRandomPersec sets the value of ReadOperationsRandomPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyReadOperationsRandomPersec(value uint32) (err error) {
	return instance.SetProperty("ReadOperationsRandomPersec", (value))
}

// GetReadOperationsRandomPersec gets the value of ReadOperationsRandomPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyReadOperationsRandomPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadOperationsRandomPersec")
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

// SetReadPacketsPersec sets the value of ReadPacketsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyReadPacketsPersec(value uint32) (err error) {
	return instance.SetProperty("ReadPacketsPersec", (value))
}

// GetReadPacketsPersec gets the value of ReadPacketsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyReadPacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadPacketsPersec")
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

// SetReadPacketsSmallPersec sets the value of ReadPacketsSmallPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyReadPacketsSmallPersec(value uint32) (err error) {
	return instance.SetProperty("ReadPacketsSmallPersec", (value))
}

// GetReadPacketsSmallPersec gets the value of ReadPacketsSmallPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyReadPacketsSmallPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadPacketsSmallPersec")
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

// SetReadsDeniedPersec sets the value of ReadsDeniedPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyReadsDeniedPersec(value uint32) (err error) {
	return instance.SetProperty("ReadsDeniedPersec", (value))
}

// GetReadsDeniedPersec gets the value of ReadsDeniedPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyReadsDeniedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadsDeniedPersec")
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

// SetReadsLargePersec sets the value of ReadsLargePersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyReadsLargePersec(value uint32) (err error) {
	return instance.SetProperty("ReadsLargePersec", (value))
}

// GetReadsLargePersec gets the value of ReadsLargePersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyReadsLargePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReadsLargePersec")
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

// SetServerDisconnects sets the value of ServerDisconnects for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyServerDisconnects(value uint32) (err error) {
	return instance.SetProperty("ServerDisconnects", (value))
}

// GetServerDisconnects gets the value of ServerDisconnects for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyServerDisconnects() (value uint32, err error) {
	retValue, err := instance.GetProperty("ServerDisconnects")
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

// SetServerReconnects sets the value of ServerReconnects for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyServerReconnects(value uint32) (err error) {
	return instance.SetProperty("ServerReconnects", (value))
}

// GetServerReconnects gets the value of ServerReconnects for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyServerReconnects() (value uint32, err error) {
	retValue, err := instance.GetProperty("ServerReconnects")
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

// SetServerSessions sets the value of ServerSessions for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyServerSessions(value uint32) (err error) {
	return instance.SetProperty("ServerSessions", (value))
}

// GetServerSessions gets the value of ServerSessions for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyServerSessions() (value uint32, err error) {
	retValue, err := instance.GetProperty("ServerSessions")
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

// SetServerSessionsHung sets the value of ServerSessionsHung for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyServerSessionsHung(value uint32) (err error) {
	return instance.SetProperty("ServerSessionsHung", (value))
}

// GetServerSessionsHung gets the value of ServerSessionsHung for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyServerSessionsHung() (value uint32, err error) {
	retValue, err := instance.GetProperty("ServerSessionsHung")
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

// SetWriteBytesCachePersec sets the value of WriteBytesCachePersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyWriteBytesCachePersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesCachePersec", (value))
}

// GetWriteBytesCachePersec gets the value of WriteBytesCachePersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyWriteBytesCachePersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBytesCachePersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetWriteBytesNetworkPersec sets the value of WriteBytesNetworkPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyWriteBytesNetworkPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesNetworkPersec", (value))
}

// GetWriteBytesNetworkPersec gets the value of WriteBytesNetworkPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyWriteBytesNetworkPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBytesNetworkPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetWriteBytesNonPagingPersec sets the value of WriteBytesNonPagingPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyWriteBytesNonPagingPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesNonPagingPersec", (value))
}

// GetWriteBytesNonPagingPersec gets the value of WriteBytesNonPagingPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyWriteBytesNonPagingPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBytesNonPagingPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetWriteBytesPagingPersec sets the value of WriteBytesPagingPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyWriteBytesPagingPersec(value uint64) (err error) {
	return instance.SetProperty("WriteBytesPagingPersec", (value))
}

// GetWriteBytesPagingPersec gets the value of WriteBytesPagingPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyWriteBytesPagingPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("WriteBytesPagingPersec")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(uint64)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " uint64 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = uint64(valuetmp)

	return
}

// SetWriteOperationsRandomPersec sets the value of WriteOperationsRandomPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyWriteOperationsRandomPersec(value uint32) (err error) {
	return instance.SetProperty("WriteOperationsRandomPersec", (value))
}

// GetWriteOperationsRandomPersec gets the value of WriteOperationsRandomPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyWriteOperationsRandomPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WriteOperationsRandomPersec")
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

// SetWritePacketsPersec sets the value of WritePacketsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyWritePacketsPersec(value uint32) (err error) {
	return instance.SetProperty("WritePacketsPersec", (value))
}

// GetWritePacketsPersec gets the value of WritePacketsPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyWritePacketsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WritePacketsPersec")
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

// SetWritePacketsSmallPersec sets the value of WritePacketsSmallPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyWritePacketsSmallPersec(value uint32) (err error) {
	return instance.SetProperty("WritePacketsSmallPersec", (value))
}

// GetWritePacketsSmallPersec gets the value of WritePacketsSmallPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyWritePacketsSmallPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WritePacketsSmallPersec")
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

// SetWritesDeniedPersec sets the value of WritesDeniedPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyWritesDeniedPersec(value uint32) (err error) {
	return instance.SetProperty("WritesDeniedPersec", (value))
}

// GetWritesDeniedPersec gets the value of WritesDeniedPersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyWritesDeniedPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WritesDeniedPersec")
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

// SetWritesLargePersec sets the value of WritesLargePersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) SetPropertyWritesLargePersec(value uint32) (err error) {
	return instance.SetProperty("WritesLargePersec", (value))
}

// GetWritesLargePersec gets the value of WritesLargePersec for the instance
func (instance *Win32_PerfRawData_PerfNet_Redirector) GetPropertyWritesLargePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WritesLargePersec")
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

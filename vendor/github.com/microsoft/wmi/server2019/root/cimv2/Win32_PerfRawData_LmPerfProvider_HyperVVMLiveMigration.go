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

// Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration struct
type Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration struct {
	*Win32_PerfRawData

	//
	CompressorBytestobeCompressed uint64

	//
	CompressorCompressedBytesSent uint64

	//
	CompressorCompressedBytesSentPersec uint64

	//
	CompressorEnabledThreads uint64

	//
	CompressorMaximumThreads uint64

	//
	MemoryWalkerBytesReadPersec uint64

	//
	MemoryWalkerBytesSentforCompression uint64

	//
	MemoryWalkerBytesSentforCompressionPersec uint64

	//
	MemoryWalkerMaximumThreads uint64

	//
	MemoryWalkerUncompressedBytesSent uint64

	//
	MemoryWalkerUncompressedBytesSentPersec uint64

	//
	ReceiverBytesPendingDecompression uint64

	//
	ReceiverBytesPendingWrite uint64

	//
	ReceiverBytesWrittenPersec uint64

	//
	ReceiverCompressedBytesReceivedPersec uint64

	//
	ReceiverDecompressedBytesPersec uint64

	//
	ReceiverMaximumThreadpoolThreadCount uint64

	//
	ReceiverUncompressedBytesReceivedPersec uint64

	//
	SMBTransportBytesSent uint64

	//
	SMBTransportBytesSentPersec uint64

	//
	SMBTransportPendingSendBytes uint64

	//
	SMBTransportPendingSendCount uint64

	//
	TCPTransportBytesPendingProcessing uint64

	//
	TCPTransportBytesPendingSend uint64

	//
	TCPTransportBytesReceivedPersec uint64

	//
	TCPTransportBytesSentPersec uint64

	//
	TCPTransportPendingSendCount uint64

	//
	TCPTransportPostedReceiveBufferCount uint64

	//
	TCPTransportTotalbuffercount uint64

	//
	TransferpassCPUCap uint64

	//
	TransferpassDirtyPageCount uint64

	//
	TransferPassIsblackout uint64

	//
	TransferPassNumber uint64
}

func NewWin32_PerfRawData_LmPerfProvider_HyperVVMLiveMigrationEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_LmPerfProvider_HyperVVMLiveMigrationEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetCompressorBytestobeCompressed sets the value of CompressorBytestobeCompressed for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyCompressorBytestobeCompressed(value uint64) (err error) {
	return instance.SetProperty("CompressorBytestobeCompressed", (value))
}

// GetCompressorBytestobeCompressed gets the value of CompressorBytestobeCompressed for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyCompressorBytestobeCompressed() (value uint64, err error) {
	retValue, err := instance.GetProperty("CompressorBytestobeCompressed")
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

// SetCompressorCompressedBytesSent sets the value of CompressorCompressedBytesSent for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyCompressorCompressedBytesSent(value uint64) (err error) {
	return instance.SetProperty("CompressorCompressedBytesSent", (value))
}

// GetCompressorCompressedBytesSent gets the value of CompressorCompressedBytesSent for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyCompressorCompressedBytesSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("CompressorCompressedBytesSent")
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

// SetCompressorCompressedBytesSentPersec sets the value of CompressorCompressedBytesSentPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyCompressorCompressedBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("CompressorCompressedBytesSentPersec", (value))
}

// GetCompressorCompressedBytesSentPersec gets the value of CompressorCompressedBytesSentPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyCompressorCompressedBytesSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("CompressorCompressedBytesSentPersec")
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

// SetCompressorEnabledThreads sets the value of CompressorEnabledThreads for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyCompressorEnabledThreads(value uint64) (err error) {
	return instance.SetProperty("CompressorEnabledThreads", (value))
}

// GetCompressorEnabledThreads gets the value of CompressorEnabledThreads for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyCompressorEnabledThreads() (value uint64, err error) {
	retValue, err := instance.GetProperty("CompressorEnabledThreads")
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

// SetCompressorMaximumThreads sets the value of CompressorMaximumThreads for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyCompressorMaximumThreads(value uint64) (err error) {
	return instance.SetProperty("CompressorMaximumThreads", (value))
}

// GetCompressorMaximumThreads gets the value of CompressorMaximumThreads for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyCompressorMaximumThreads() (value uint64, err error) {
	retValue, err := instance.GetProperty("CompressorMaximumThreads")
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

// SetMemoryWalkerBytesReadPersec sets the value of MemoryWalkerBytesReadPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyMemoryWalkerBytesReadPersec(value uint64) (err error) {
	return instance.SetProperty("MemoryWalkerBytesReadPersec", (value))
}

// GetMemoryWalkerBytesReadPersec gets the value of MemoryWalkerBytesReadPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyMemoryWalkerBytesReadPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("MemoryWalkerBytesReadPersec")
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

// SetMemoryWalkerBytesSentforCompression sets the value of MemoryWalkerBytesSentforCompression for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyMemoryWalkerBytesSentforCompression(value uint64) (err error) {
	return instance.SetProperty("MemoryWalkerBytesSentforCompression", (value))
}

// GetMemoryWalkerBytesSentforCompression gets the value of MemoryWalkerBytesSentforCompression for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyMemoryWalkerBytesSentforCompression() (value uint64, err error) {
	retValue, err := instance.GetProperty("MemoryWalkerBytesSentforCompression")
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

// SetMemoryWalkerBytesSentforCompressionPersec sets the value of MemoryWalkerBytesSentforCompressionPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyMemoryWalkerBytesSentforCompressionPersec(value uint64) (err error) {
	return instance.SetProperty("MemoryWalkerBytesSentforCompressionPersec", (value))
}

// GetMemoryWalkerBytesSentforCompressionPersec gets the value of MemoryWalkerBytesSentforCompressionPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyMemoryWalkerBytesSentforCompressionPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("MemoryWalkerBytesSentforCompressionPersec")
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

// SetMemoryWalkerMaximumThreads sets the value of MemoryWalkerMaximumThreads for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyMemoryWalkerMaximumThreads(value uint64) (err error) {
	return instance.SetProperty("MemoryWalkerMaximumThreads", (value))
}

// GetMemoryWalkerMaximumThreads gets the value of MemoryWalkerMaximumThreads for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyMemoryWalkerMaximumThreads() (value uint64, err error) {
	retValue, err := instance.GetProperty("MemoryWalkerMaximumThreads")
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

// SetMemoryWalkerUncompressedBytesSent sets the value of MemoryWalkerUncompressedBytesSent for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyMemoryWalkerUncompressedBytesSent(value uint64) (err error) {
	return instance.SetProperty("MemoryWalkerUncompressedBytesSent", (value))
}

// GetMemoryWalkerUncompressedBytesSent gets the value of MemoryWalkerUncompressedBytesSent for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyMemoryWalkerUncompressedBytesSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("MemoryWalkerUncompressedBytesSent")
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

// SetMemoryWalkerUncompressedBytesSentPersec sets the value of MemoryWalkerUncompressedBytesSentPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyMemoryWalkerUncompressedBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("MemoryWalkerUncompressedBytesSentPersec", (value))
}

// GetMemoryWalkerUncompressedBytesSentPersec gets the value of MemoryWalkerUncompressedBytesSentPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyMemoryWalkerUncompressedBytesSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("MemoryWalkerUncompressedBytesSentPersec")
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

// SetReceiverBytesPendingDecompression sets the value of ReceiverBytesPendingDecompression for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyReceiverBytesPendingDecompression(value uint64) (err error) {
	return instance.SetProperty("ReceiverBytesPendingDecompression", (value))
}

// GetReceiverBytesPendingDecompression gets the value of ReceiverBytesPendingDecompression for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyReceiverBytesPendingDecompression() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiverBytesPendingDecompression")
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

// SetReceiverBytesPendingWrite sets the value of ReceiverBytesPendingWrite for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyReceiverBytesPendingWrite(value uint64) (err error) {
	return instance.SetProperty("ReceiverBytesPendingWrite", (value))
}

// GetReceiverBytesPendingWrite gets the value of ReceiverBytesPendingWrite for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyReceiverBytesPendingWrite() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiverBytesPendingWrite")
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

// SetReceiverBytesWrittenPersec sets the value of ReceiverBytesWrittenPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyReceiverBytesWrittenPersec(value uint64) (err error) {
	return instance.SetProperty("ReceiverBytesWrittenPersec", (value))
}

// GetReceiverBytesWrittenPersec gets the value of ReceiverBytesWrittenPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyReceiverBytesWrittenPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiverBytesWrittenPersec")
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

// SetReceiverCompressedBytesReceivedPersec sets the value of ReceiverCompressedBytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyReceiverCompressedBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("ReceiverCompressedBytesReceivedPersec", (value))
}

// GetReceiverCompressedBytesReceivedPersec gets the value of ReceiverCompressedBytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyReceiverCompressedBytesReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiverCompressedBytesReceivedPersec")
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

// SetReceiverDecompressedBytesPersec sets the value of ReceiverDecompressedBytesPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyReceiverDecompressedBytesPersec(value uint64) (err error) {
	return instance.SetProperty("ReceiverDecompressedBytesPersec", (value))
}

// GetReceiverDecompressedBytesPersec gets the value of ReceiverDecompressedBytesPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyReceiverDecompressedBytesPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiverDecompressedBytesPersec")
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

// SetReceiverMaximumThreadpoolThreadCount sets the value of ReceiverMaximumThreadpoolThreadCount for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyReceiverMaximumThreadpoolThreadCount(value uint64) (err error) {
	return instance.SetProperty("ReceiverMaximumThreadpoolThreadCount", (value))
}

// GetReceiverMaximumThreadpoolThreadCount gets the value of ReceiverMaximumThreadpoolThreadCount for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyReceiverMaximumThreadpoolThreadCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiverMaximumThreadpoolThreadCount")
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

// SetReceiverUncompressedBytesReceivedPersec sets the value of ReceiverUncompressedBytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyReceiverUncompressedBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("ReceiverUncompressedBytesReceivedPersec", (value))
}

// GetReceiverUncompressedBytesReceivedPersec gets the value of ReceiverUncompressedBytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyReceiverUncompressedBytesReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("ReceiverUncompressedBytesReceivedPersec")
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

// SetSMBTransportBytesSent sets the value of SMBTransportBytesSent for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertySMBTransportBytesSent(value uint64) (err error) {
	return instance.SetProperty("SMBTransportBytesSent", (value))
}

// GetSMBTransportBytesSent gets the value of SMBTransportBytesSent for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertySMBTransportBytesSent() (value uint64, err error) {
	retValue, err := instance.GetProperty("SMBTransportBytesSent")
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

// SetSMBTransportBytesSentPersec sets the value of SMBTransportBytesSentPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertySMBTransportBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("SMBTransportBytesSentPersec", (value))
}

// GetSMBTransportBytesSentPersec gets the value of SMBTransportBytesSentPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertySMBTransportBytesSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("SMBTransportBytesSentPersec")
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

// SetSMBTransportPendingSendBytes sets the value of SMBTransportPendingSendBytes for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertySMBTransportPendingSendBytes(value uint64) (err error) {
	return instance.SetProperty("SMBTransportPendingSendBytes", (value))
}

// GetSMBTransportPendingSendBytes gets the value of SMBTransportPendingSendBytes for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertySMBTransportPendingSendBytes() (value uint64, err error) {
	retValue, err := instance.GetProperty("SMBTransportPendingSendBytes")
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

// SetSMBTransportPendingSendCount sets the value of SMBTransportPendingSendCount for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertySMBTransportPendingSendCount(value uint64) (err error) {
	return instance.SetProperty("SMBTransportPendingSendCount", (value))
}

// GetSMBTransportPendingSendCount gets the value of SMBTransportPendingSendCount for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertySMBTransportPendingSendCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("SMBTransportPendingSendCount")
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

// SetTCPTransportBytesPendingProcessing sets the value of TCPTransportBytesPendingProcessing for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyTCPTransportBytesPendingProcessing(value uint64) (err error) {
	return instance.SetProperty("TCPTransportBytesPendingProcessing", (value))
}

// GetTCPTransportBytesPendingProcessing gets the value of TCPTransportBytesPendingProcessing for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyTCPTransportBytesPendingProcessing() (value uint64, err error) {
	retValue, err := instance.GetProperty("TCPTransportBytesPendingProcessing")
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

// SetTCPTransportBytesPendingSend sets the value of TCPTransportBytesPendingSend for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyTCPTransportBytesPendingSend(value uint64) (err error) {
	return instance.SetProperty("TCPTransportBytesPendingSend", (value))
}

// GetTCPTransportBytesPendingSend gets the value of TCPTransportBytesPendingSend for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyTCPTransportBytesPendingSend() (value uint64, err error) {
	retValue, err := instance.GetProperty("TCPTransportBytesPendingSend")
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

// SetTCPTransportBytesReceivedPersec sets the value of TCPTransportBytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyTCPTransportBytesReceivedPersec(value uint64) (err error) {
	return instance.SetProperty("TCPTransportBytesReceivedPersec", (value))
}

// GetTCPTransportBytesReceivedPersec gets the value of TCPTransportBytesReceivedPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyTCPTransportBytesReceivedPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("TCPTransportBytesReceivedPersec")
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

// SetTCPTransportBytesSentPersec sets the value of TCPTransportBytesSentPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyTCPTransportBytesSentPersec(value uint64) (err error) {
	return instance.SetProperty("TCPTransportBytesSentPersec", (value))
}

// GetTCPTransportBytesSentPersec gets the value of TCPTransportBytesSentPersec for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyTCPTransportBytesSentPersec() (value uint64, err error) {
	retValue, err := instance.GetProperty("TCPTransportBytesSentPersec")
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

// SetTCPTransportPendingSendCount sets the value of TCPTransportPendingSendCount for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyTCPTransportPendingSendCount(value uint64) (err error) {
	return instance.SetProperty("TCPTransportPendingSendCount", (value))
}

// GetTCPTransportPendingSendCount gets the value of TCPTransportPendingSendCount for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyTCPTransportPendingSendCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("TCPTransportPendingSendCount")
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

// SetTCPTransportPostedReceiveBufferCount sets the value of TCPTransportPostedReceiveBufferCount for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyTCPTransportPostedReceiveBufferCount(value uint64) (err error) {
	return instance.SetProperty("TCPTransportPostedReceiveBufferCount", (value))
}

// GetTCPTransportPostedReceiveBufferCount gets the value of TCPTransportPostedReceiveBufferCount for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyTCPTransportPostedReceiveBufferCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("TCPTransportPostedReceiveBufferCount")
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

// SetTCPTransportTotalbuffercount sets the value of TCPTransportTotalbuffercount for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyTCPTransportTotalbuffercount(value uint64) (err error) {
	return instance.SetProperty("TCPTransportTotalbuffercount", (value))
}

// GetTCPTransportTotalbuffercount gets the value of TCPTransportTotalbuffercount for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyTCPTransportTotalbuffercount() (value uint64, err error) {
	retValue, err := instance.GetProperty("TCPTransportTotalbuffercount")
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

// SetTransferpassCPUCap sets the value of TransferpassCPUCap for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyTransferpassCPUCap(value uint64) (err error) {
	return instance.SetProperty("TransferpassCPUCap", (value))
}

// GetTransferpassCPUCap gets the value of TransferpassCPUCap for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyTransferpassCPUCap() (value uint64, err error) {
	retValue, err := instance.GetProperty("TransferpassCPUCap")
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

// SetTransferpassDirtyPageCount sets the value of TransferpassDirtyPageCount for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyTransferpassDirtyPageCount(value uint64) (err error) {
	return instance.SetProperty("TransferpassDirtyPageCount", (value))
}

// GetTransferpassDirtyPageCount gets the value of TransferpassDirtyPageCount for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyTransferpassDirtyPageCount() (value uint64, err error) {
	retValue, err := instance.GetProperty("TransferpassDirtyPageCount")
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

// SetTransferPassIsblackout sets the value of TransferPassIsblackout for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyTransferPassIsblackout(value uint64) (err error) {
	return instance.SetProperty("TransferPassIsblackout", (value))
}

// GetTransferPassIsblackout gets the value of TransferPassIsblackout for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyTransferPassIsblackout() (value uint64, err error) {
	retValue, err := instance.GetProperty("TransferPassIsblackout")
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

// SetTransferPassNumber sets the value of TransferPassNumber for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) SetPropertyTransferPassNumber(value uint64) (err error) {
	return instance.SetProperty("TransferPassNumber", (value))
}

// GetTransferPassNumber gets the value of TransferPassNumber for the instance
func (instance *Win32_PerfRawData_LmPerfProvider_HyperVVMLiveMigration) GetPropertyTransferPassNumber() (value uint64, err error) {
	retValue, err := instance.GetProperty("TransferPassNumber")
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

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

// Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters struct
type Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters struct {
	*Win32_PerfFormattedData

	//
	CQOverflows uint64

	//
	RequesterCQEErrors uint64

	//
	RequesterInvalidRequestErrors uint64

	//
	RequesterLengthErrors uint64

	//
	RequesterOutoforderSequenceNAK uint64

	//
	RequesterProtectionErrors uint64

	//
	RequesterQPOperationErrors uint64

	//
	RequesterRemoteAccessErrors uint64

	//
	RequesterRemoteOperationErrors uint64

	//
	RequesterRNRNAK uint64

	//
	RequesterRNRNAKRetriesExceededErrors uint64

	//
	RequesterTimeoutReceived uint64

	//
	RequesterTransportRetriesExceededErrors uint64

	//
	ResponderCQEErrors uint64

	//
	ResponderDuplicateRequestReceived uint64

	//
	ResponderInvalidRequestErrors uint64

	//
	ResponderLengthErrors uint64

	//
	ResponderOutoforderSequenceReceived uint64

	//
	ResponderProtectionErrors uint64

	//
	ResponderQPOperationErrors uint64

	//
	ResponderRemoteAccessErrors uint64

	//
	ResponderRNRNAK uint64

	//
	TXRingIsFullPackets uint64
}

func NewWin32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCountersEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters{
		Win32_PerfFormattedData: tmp,
	}
	return
}

func NewWin32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCountersEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters, err error) {
	tmp, err := NewWin32_PerfFormattedDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters{
		Win32_PerfFormattedData: tmp,
	}
	return
}

// SetCQOverflows sets the value of CQOverflows for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyCQOverflows(value uint64) (err error) {
	return instance.SetProperty("CQOverflows", (value))
}

// GetCQOverflows gets the value of CQOverflows for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyCQOverflows() (value uint64, err error) {
	retValue, err := instance.GetProperty("CQOverflows")
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

// SetRequesterCQEErrors sets the value of RequesterCQEErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyRequesterCQEErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterCQEErrors", (value))
}

// GetRequesterCQEErrors gets the value of RequesterCQEErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyRequesterCQEErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterCQEErrors")
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

// SetRequesterInvalidRequestErrors sets the value of RequesterInvalidRequestErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyRequesterInvalidRequestErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterInvalidRequestErrors", (value))
}

// GetRequesterInvalidRequestErrors gets the value of RequesterInvalidRequestErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyRequesterInvalidRequestErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterInvalidRequestErrors")
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

// SetRequesterLengthErrors sets the value of RequesterLengthErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyRequesterLengthErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterLengthErrors", (value))
}

// GetRequesterLengthErrors gets the value of RequesterLengthErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyRequesterLengthErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterLengthErrors")
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

// SetRequesterOutoforderSequenceNAK sets the value of RequesterOutoforderSequenceNAK for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyRequesterOutoforderSequenceNAK(value uint64) (err error) {
	return instance.SetProperty("RequesterOutoforderSequenceNAK", (value))
}

// GetRequesterOutoforderSequenceNAK gets the value of RequesterOutoforderSequenceNAK for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyRequesterOutoforderSequenceNAK() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterOutoforderSequenceNAK")
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

// SetRequesterProtectionErrors sets the value of RequesterProtectionErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyRequesterProtectionErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterProtectionErrors", (value))
}

// GetRequesterProtectionErrors gets the value of RequesterProtectionErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyRequesterProtectionErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterProtectionErrors")
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

// SetRequesterQPOperationErrors sets the value of RequesterQPOperationErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyRequesterQPOperationErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterQPOperationErrors", (value))
}

// GetRequesterQPOperationErrors gets the value of RequesterQPOperationErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyRequesterQPOperationErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterQPOperationErrors")
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

// SetRequesterRemoteAccessErrors sets the value of RequesterRemoteAccessErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyRequesterRemoteAccessErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterRemoteAccessErrors", (value))
}

// GetRequesterRemoteAccessErrors gets the value of RequesterRemoteAccessErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyRequesterRemoteAccessErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterRemoteAccessErrors")
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

// SetRequesterRemoteOperationErrors sets the value of RequesterRemoteOperationErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyRequesterRemoteOperationErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterRemoteOperationErrors", (value))
}

// GetRequesterRemoteOperationErrors gets the value of RequesterRemoteOperationErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyRequesterRemoteOperationErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterRemoteOperationErrors")
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

// SetRequesterRNRNAK sets the value of RequesterRNRNAK for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyRequesterRNRNAK(value uint64) (err error) {
	return instance.SetProperty("RequesterRNRNAK", (value))
}

// GetRequesterRNRNAK gets the value of RequesterRNRNAK for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyRequesterRNRNAK() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterRNRNAK")
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

// SetRequesterRNRNAKRetriesExceededErrors sets the value of RequesterRNRNAKRetriesExceededErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyRequesterRNRNAKRetriesExceededErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterRNRNAKRetriesExceededErrors", (value))
}

// GetRequesterRNRNAKRetriesExceededErrors gets the value of RequesterRNRNAKRetriesExceededErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyRequesterRNRNAKRetriesExceededErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterRNRNAKRetriesExceededErrors")
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

// SetRequesterTimeoutReceived sets the value of RequesterTimeoutReceived for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyRequesterTimeoutReceived(value uint64) (err error) {
	return instance.SetProperty("RequesterTimeoutReceived", (value))
}

// GetRequesterTimeoutReceived gets the value of RequesterTimeoutReceived for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyRequesterTimeoutReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterTimeoutReceived")
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

// SetRequesterTransportRetriesExceededErrors sets the value of RequesterTransportRetriesExceededErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyRequesterTransportRetriesExceededErrors(value uint64) (err error) {
	return instance.SetProperty("RequesterTransportRetriesExceededErrors", (value))
}

// GetRequesterTransportRetriesExceededErrors gets the value of RequesterTransportRetriesExceededErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyRequesterTransportRetriesExceededErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("RequesterTransportRetriesExceededErrors")
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

// SetResponderCQEErrors sets the value of ResponderCQEErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyResponderCQEErrors(value uint64) (err error) {
	return instance.SetProperty("ResponderCQEErrors", (value))
}

// GetResponderCQEErrors gets the value of ResponderCQEErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyResponderCQEErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderCQEErrors")
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

// SetResponderDuplicateRequestReceived sets the value of ResponderDuplicateRequestReceived for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyResponderDuplicateRequestReceived(value uint64) (err error) {
	return instance.SetProperty("ResponderDuplicateRequestReceived", (value))
}

// GetResponderDuplicateRequestReceived gets the value of ResponderDuplicateRequestReceived for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyResponderDuplicateRequestReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderDuplicateRequestReceived")
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

// SetResponderInvalidRequestErrors sets the value of ResponderInvalidRequestErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyResponderInvalidRequestErrors(value uint64) (err error) {
	return instance.SetProperty("ResponderInvalidRequestErrors", (value))
}

// GetResponderInvalidRequestErrors gets the value of ResponderInvalidRequestErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyResponderInvalidRequestErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderInvalidRequestErrors")
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

// SetResponderLengthErrors sets the value of ResponderLengthErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyResponderLengthErrors(value uint64) (err error) {
	return instance.SetProperty("ResponderLengthErrors", (value))
}

// GetResponderLengthErrors gets the value of ResponderLengthErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyResponderLengthErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderLengthErrors")
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

// SetResponderOutoforderSequenceReceived sets the value of ResponderOutoforderSequenceReceived for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyResponderOutoforderSequenceReceived(value uint64) (err error) {
	return instance.SetProperty("ResponderOutoforderSequenceReceived", (value))
}

// GetResponderOutoforderSequenceReceived gets the value of ResponderOutoforderSequenceReceived for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyResponderOutoforderSequenceReceived() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderOutoforderSequenceReceived")
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

// SetResponderProtectionErrors sets the value of ResponderProtectionErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyResponderProtectionErrors(value uint64) (err error) {
	return instance.SetProperty("ResponderProtectionErrors", (value))
}

// GetResponderProtectionErrors gets the value of ResponderProtectionErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyResponderProtectionErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderProtectionErrors")
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

// SetResponderQPOperationErrors sets the value of ResponderQPOperationErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyResponderQPOperationErrors(value uint64) (err error) {
	return instance.SetProperty("ResponderQPOperationErrors", (value))
}

// GetResponderQPOperationErrors gets the value of ResponderQPOperationErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyResponderQPOperationErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderQPOperationErrors")
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

// SetResponderRemoteAccessErrors sets the value of ResponderRemoteAccessErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyResponderRemoteAccessErrors(value uint64) (err error) {
	return instance.SetProperty("ResponderRemoteAccessErrors", (value))
}

// GetResponderRemoteAccessErrors gets the value of ResponderRemoteAccessErrors for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyResponderRemoteAccessErrors() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderRemoteAccessErrors")
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

// SetResponderRNRNAK sets the value of ResponderRNRNAK for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyResponderRNRNAK(value uint64) (err error) {
	return instance.SetProperty("ResponderRNRNAK", (value))
}

// GetResponderRNRNAK gets the value of ResponderRNRNAK for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyResponderRNRNAK() (value uint64, err error) {
	retValue, err := instance.GetProperty("ResponderRNRNAK")
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

// SetTXRingIsFullPackets sets the value of TXRingIsFullPackets for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) SetPropertyTXRingIsFullPackets(value uint64) (err error) {
	return instance.SetProperty("TXRingIsFullPackets", (value))
}

// GetTXRingIsFullPackets gets the value of TXRingIsFullPackets for the instance
func (instance *Win32_PerfFormattedData_Mlx4IBProvider_MellanoxIBAdapterDiagnosticCounters) GetPropertyTXRingIsFullPackets() (value uint64, err error) {
	retValue, err := instance.GetProperty("TXRingIsFullPackets")
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

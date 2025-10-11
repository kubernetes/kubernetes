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

// Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters struct
type Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters struct {
	*Win32_PerfRawData

	//
	ArrivedRDMACNPs uint64

	//
	CPUMEMpages4KmappedbyTPTforCQ uint32

	//
	CPUMEMpages4KmappedbyTPTforEQ uint32

	//
	CPUMEMpages4KmappedbyTPTforMR uint32

	//
	CPUMEMpages4KmappedbyTPTforQP uint32

	//
	CQMissPersec uint32

	//
	Currentqpsinerrorstate uint32

	//
	Currentqpsinlimitedstate uint32

	//
	Dcqcnreactionpointnewqprejectevents uint32

	//
	Dcqcnreactionpointnewqpshapedevents uint32

	//
	Dcqcnreactionpointqprateupdateevents uint32

	//
	Dcqcnreactionpointqpscheduleddelayedevents uint32

	//
	Dcqcnreactionpointqpschedulednotshapedevents uint32

	//
	Dcqcnreactionpointqpscheduledpermitedevents uint32

	//
	EQMissPersec uint32

	//
	ExternalBlueflamehitPersec uint32

	//
	ExternalBlueflameReplacePersec uint32

	//
	ExternalDoorbellDropPersec uint32

	//
	ExternalDoorbellPushPersec uint32

	//
	InternalProcessor0MaximumLatency uint32

	//
	InternalProcessor1MaximumLatency uint32

	//
	InternalProcessor2MaximumLatency uint32

	//
	InternalProcessor3MaximumLatency uint32

	//
	Internalprocessorexecutedcommands uint32

	//
	LastRestransmittedQP uint32

	//
	Maximumqpsinlimitedstate uint32

	//
	MPTentriesusedforCQ uint32

	//
	MPTentriesusedforEQ uint32

	//
	MPTentriesusedforMR uint32

	//
	MPTentriesusedforQP uint32

	//
	MPTMissPersec uint32

	//
	MTTentriesusedforCQ uint32

	//
	MTTentriesusedforEQ uint32

	//
	MTTentriesusedforMR uint32

	//
	MTTentriesusedforQP uint32

	//
	MTTMissPersec uint32

	//
	NoWQEDropsPersec uint32

	//
	Packetsdiscardedduetoinvalidqp uint64

	//
	PCIBackpressurePersec uint32

	//
	Qppriorityupdateflowevents uint32

	//
	ReceiveWQEcachehitPersec uint32

	//
	ReceiveWQEcachelookupPersec uint32

	//
	RQMissPersec uint32

	//
	ScatterBackpressurePersec uint32

	//
	SQMissPersec uint32

	//
	SteeringQPCBackpressurePersec uint32

	//
	Totalqpsinlimitedstate uint32

	//
	Transmissionenginehangevents uint32

	//
	WQEfetchPerAtomicBackpressurePersec uint32
}

func NewWin32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCountersEx1(instance *cim.WmiInstance) (newInstance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters, err error) {
	tmp, err := NewWin32_PerfRawDataEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters{
		Win32_PerfRawData: tmp,
	}
	return
}

func NewWin32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCountersEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters, err error) {
	tmp, err := NewWin32_PerfRawDataEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters{
		Win32_PerfRawData: tmp,
	}
	return
}

// SetArrivedRDMACNPs sets the value of ArrivedRDMACNPs for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyArrivedRDMACNPs(value uint64) (err error) {
	return instance.SetProperty("ArrivedRDMACNPs", (value))
}

// GetArrivedRDMACNPs gets the value of ArrivedRDMACNPs for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyArrivedRDMACNPs() (value uint64, err error) {
	retValue, err := instance.GetProperty("ArrivedRDMACNPs")
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

// SetCPUMEMpages4KmappedbyTPTforCQ sets the value of CPUMEMpages4KmappedbyTPTforCQ for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyCPUMEMpages4KmappedbyTPTforCQ(value uint32) (err error) {
	return instance.SetProperty("CPUMEMpages4KmappedbyTPTforCQ", (value))
}

// GetCPUMEMpages4KmappedbyTPTforCQ gets the value of CPUMEMpages4KmappedbyTPTforCQ for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyCPUMEMpages4KmappedbyTPTforCQ() (value uint32, err error) {
	retValue, err := instance.GetProperty("CPUMEMpages4KmappedbyTPTforCQ")
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

// SetCPUMEMpages4KmappedbyTPTforEQ sets the value of CPUMEMpages4KmappedbyTPTforEQ for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyCPUMEMpages4KmappedbyTPTforEQ(value uint32) (err error) {
	return instance.SetProperty("CPUMEMpages4KmappedbyTPTforEQ", (value))
}

// GetCPUMEMpages4KmappedbyTPTforEQ gets the value of CPUMEMpages4KmappedbyTPTforEQ for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyCPUMEMpages4KmappedbyTPTforEQ() (value uint32, err error) {
	retValue, err := instance.GetProperty("CPUMEMpages4KmappedbyTPTforEQ")
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

// SetCPUMEMpages4KmappedbyTPTforMR sets the value of CPUMEMpages4KmappedbyTPTforMR for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyCPUMEMpages4KmappedbyTPTforMR(value uint32) (err error) {
	return instance.SetProperty("CPUMEMpages4KmappedbyTPTforMR", (value))
}

// GetCPUMEMpages4KmappedbyTPTforMR gets the value of CPUMEMpages4KmappedbyTPTforMR for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyCPUMEMpages4KmappedbyTPTforMR() (value uint32, err error) {
	retValue, err := instance.GetProperty("CPUMEMpages4KmappedbyTPTforMR")
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

// SetCPUMEMpages4KmappedbyTPTforQP sets the value of CPUMEMpages4KmappedbyTPTforQP for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyCPUMEMpages4KmappedbyTPTforQP(value uint32) (err error) {
	return instance.SetProperty("CPUMEMpages4KmappedbyTPTforQP", (value))
}

// GetCPUMEMpages4KmappedbyTPTforQP gets the value of CPUMEMpages4KmappedbyTPTforQP for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyCPUMEMpages4KmappedbyTPTforQP() (value uint32, err error) {
	retValue, err := instance.GetProperty("CPUMEMpages4KmappedbyTPTforQP")
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

// SetCQMissPersec sets the value of CQMissPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyCQMissPersec(value uint32) (err error) {
	return instance.SetProperty("CQMissPersec", (value))
}

// GetCQMissPersec gets the value of CQMissPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyCQMissPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("CQMissPersec")
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

// SetCurrentqpsinerrorstate sets the value of Currentqpsinerrorstate for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyCurrentqpsinerrorstate(value uint32) (err error) {
	return instance.SetProperty("Currentqpsinerrorstate", (value))
}

// GetCurrentqpsinerrorstate gets the value of Currentqpsinerrorstate for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyCurrentqpsinerrorstate() (value uint32, err error) {
	retValue, err := instance.GetProperty("Currentqpsinerrorstate")
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

// SetCurrentqpsinlimitedstate sets the value of Currentqpsinlimitedstate for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyCurrentqpsinlimitedstate(value uint32) (err error) {
	return instance.SetProperty("Currentqpsinlimitedstate", (value))
}

// GetCurrentqpsinlimitedstate gets the value of Currentqpsinlimitedstate for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyCurrentqpsinlimitedstate() (value uint32, err error) {
	retValue, err := instance.GetProperty("Currentqpsinlimitedstate")
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

// SetDcqcnreactionpointnewqprejectevents sets the value of Dcqcnreactionpointnewqprejectevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyDcqcnreactionpointnewqprejectevents(value uint32) (err error) {
	return instance.SetProperty("Dcqcnreactionpointnewqprejectevents", (value))
}

// GetDcqcnreactionpointnewqprejectevents gets the value of Dcqcnreactionpointnewqprejectevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyDcqcnreactionpointnewqprejectevents() (value uint32, err error) {
	retValue, err := instance.GetProperty("Dcqcnreactionpointnewqprejectevents")
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

// SetDcqcnreactionpointnewqpshapedevents sets the value of Dcqcnreactionpointnewqpshapedevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyDcqcnreactionpointnewqpshapedevents(value uint32) (err error) {
	return instance.SetProperty("Dcqcnreactionpointnewqpshapedevents", (value))
}

// GetDcqcnreactionpointnewqpshapedevents gets the value of Dcqcnreactionpointnewqpshapedevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyDcqcnreactionpointnewqpshapedevents() (value uint32, err error) {
	retValue, err := instance.GetProperty("Dcqcnreactionpointnewqpshapedevents")
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

// SetDcqcnreactionpointqprateupdateevents sets the value of Dcqcnreactionpointqprateupdateevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyDcqcnreactionpointqprateupdateevents(value uint32) (err error) {
	return instance.SetProperty("Dcqcnreactionpointqprateupdateevents", (value))
}

// GetDcqcnreactionpointqprateupdateevents gets the value of Dcqcnreactionpointqprateupdateevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyDcqcnreactionpointqprateupdateevents() (value uint32, err error) {
	retValue, err := instance.GetProperty("Dcqcnreactionpointqprateupdateevents")
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

// SetDcqcnreactionpointqpscheduleddelayedevents sets the value of Dcqcnreactionpointqpscheduleddelayedevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyDcqcnreactionpointqpscheduleddelayedevents(value uint32) (err error) {
	return instance.SetProperty("Dcqcnreactionpointqpscheduleddelayedevents", (value))
}

// GetDcqcnreactionpointqpscheduleddelayedevents gets the value of Dcqcnreactionpointqpscheduleddelayedevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyDcqcnreactionpointqpscheduleddelayedevents() (value uint32, err error) {
	retValue, err := instance.GetProperty("Dcqcnreactionpointqpscheduleddelayedevents")
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

// SetDcqcnreactionpointqpschedulednotshapedevents sets the value of Dcqcnreactionpointqpschedulednotshapedevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyDcqcnreactionpointqpschedulednotshapedevents(value uint32) (err error) {
	return instance.SetProperty("Dcqcnreactionpointqpschedulednotshapedevents", (value))
}

// GetDcqcnreactionpointqpschedulednotshapedevents gets the value of Dcqcnreactionpointqpschedulednotshapedevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyDcqcnreactionpointqpschedulednotshapedevents() (value uint32, err error) {
	retValue, err := instance.GetProperty("Dcqcnreactionpointqpschedulednotshapedevents")
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

// SetDcqcnreactionpointqpscheduledpermitedevents sets the value of Dcqcnreactionpointqpscheduledpermitedevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyDcqcnreactionpointqpscheduledpermitedevents(value uint32) (err error) {
	return instance.SetProperty("Dcqcnreactionpointqpscheduledpermitedevents", (value))
}

// GetDcqcnreactionpointqpscheduledpermitedevents gets the value of Dcqcnreactionpointqpscheduledpermitedevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyDcqcnreactionpointqpscheduledpermitedevents() (value uint32, err error) {
	retValue, err := instance.GetProperty("Dcqcnreactionpointqpscheduledpermitedevents")
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

// SetEQMissPersec sets the value of EQMissPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyEQMissPersec(value uint32) (err error) {
	return instance.SetProperty("EQMissPersec", (value))
}

// GetEQMissPersec gets the value of EQMissPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyEQMissPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("EQMissPersec")
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

// SetExternalBlueflamehitPersec sets the value of ExternalBlueflamehitPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyExternalBlueflamehitPersec(value uint32) (err error) {
	return instance.SetProperty("ExternalBlueflamehitPersec", (value))
}

// GetExternalBlueflamehitPersec gets the value of ExternalBlueflamehitPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyExternalBlueflamehitPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ExternalBlueflamehitPersec")
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

// SetExternalBlueflameReplacePersec sets the value of ExternalBlueflameReplacePersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyExternalBlueflameReplacePersec(value uint32) (err error) {
	return instance.SetProperty("ExternalBlueflameReplacePersec", (value))
}

// GetExternalBlueflameReplacePersec gets the value of ExternalBlueflameReplacePersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyExternalBlueflameReplacePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ExternalBlueflameReplacePersec")
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

// SetExternalDoorbellDropPersec sets the value of ExternalDoorbellDropPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyExternalDoorbellDropPersec(value uint32) (err error) {
	return instance.SetProperty("ExternalDoorbellDropPersec", (value))
}

// GetExternalDoorbellDropPersec gets the value of ExternalDoorbellDropPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyExternalDoorbellDropPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ExternalDoorbellDropPersec")
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

// SetExternalDoorbellPushPersec sets the value of ExternalDoorbellPushPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyExternalDoorbellPushPersec(value uint32) (err error) {
	return instance.SetProperty("ExternalDoorbellPushPersec", (value))
}

// GetExternalDoorbellPushPersec gets the value of ExternalDoorbellPushPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyExternalDoorbellPushPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ExternalDoorbellPushPersec")
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

// SetInternalProcessor0MaximumLatency sets the value of InternalProcessor0MaximumLatency for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyInternalProcessor0MaximumLatency(value uint32) (err error) {
	return instance.SetProperty("InternalProcessor0MaximumLatency", (value))
}

// GetInternalProcessor0MaximumLatency gets the value of InternalProcessor0MaximumLatency for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyInternalProcessor0MaximumLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("InternalProcessor0MaximumLatency")
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

// SetInternalProcessor1MaximumLatency sets the value of InternalProcessor1MaximumLatency for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyInternalProcessor1MaximumLatency(value uint32) (err error) {
	return instance.SetProperty("InternalProcessor1MaximumLatency", (value))
}

// GetInternalProcessor1MaximumLatency gets the value of InternalProcessor1MaximumLatency for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyInternalProcessor1MaximumLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("InternalProcessor1MaximumLatency")
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

// SetInternalProcessor2MaximumLatency sets the value of InternalProcessor2MaximumLatency for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyInternalProcessor2MaximumLatency(value uint32) (err error) {
	return instance.SetProperty("InternalProcessor2MaximumLatency", (value))
}

// GetInternalProcessor2MaximumLatency gets the value of InternalProcessor2MaximumLatency for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyInternalProcessor2MaximumLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("InternalProcessor2MaximumLatency")
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

// SetInternalProcessor3MaximumLatency sets the value of InternalProcessor3MaximumLatency for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyInternalProcessor3MaximumLatency(value uint32) (err error) {
	return instance.SetProperty("InternalProcessor3MaximumLatency", (value))
}

// GetInternalProcessor3MaximumLatency gets the value of InternalProcessor3MaximumLatency for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyInternalProcessor3MaximumLatency() (value uint32, err error) {
	retValue, err := instance.GetProperty("InternalProcessor3MaximumLatency")
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

// SetInternalprocessorexecutedcommands sets the value of Internalprocessorexecutedcommands for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyInternalprocessorexecutedcommands(value uint32) (err error) {
	return instance.SetProperty("Internalprocessorexecutedcommands", (value))
}

// GetInternalprocessorexecutedcommands gets the value of Internalprocessorexecutedcommands for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyInternalprocessorexecutedcommands() (value uint32, err error) {
	retValue, err := instance.GetProperty("Internalprocessorexecutedcommands")
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

// SetLastRestransmittedQP sets the value of LastRestransmittedQP for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyLastRestransmittedQP(value uint32) (err error) {
	return instance.SetProperty("LastRestransmittedQP", (value))
}

// GetLastRestransmittedQP gets the value of LastRestransmittedQP for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyLastRestransmittedQP() (value uint32, err error) {
	retValue, err := instance.GetProperty("LastRestransmittedQP")
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

// SetMaximumqpsinlimitedstate sets the value of Maximumqpsinlimitedstate for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyMaximumqpsinlimitedstate(value uint32) (err error) {
	return instance.SetProperty("Maximumqpsinlimitedstate", (value))
}

// GetMaximumqpsinlimitedstate gets the value of Maximumqpsinlimitedstate for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyMaximumqpsinlimitedstate() (value uint32, err error) {
	retValue, err := instance.GetProperty("Maximumqpsinlimitedstate")
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

// SetMPTentriesusedforCQ sets the value of MPTentriesusedforCQ for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyMPTentriesusedforCQ(value uint32) (err error) {
	return instance.SetProperty("MPTentriesusedforCQ", (value))
}

// GetMPTentriesusedforCQ gets the value of MPTentriesusedforCQ for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyMPTentriesusedforCQ() (value uint32, err error) {
	retValue, err := instance.GetProperty("MPTentriesusedforCQ")
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

// SetMPTentriesusedforEQ sets the value of MPTentriesusedforEQ for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyMPTentriesusedforEQ(value uint32) (err error) {
	return instance.SetProperty("MPTentriesusedforEQ", (value))
}

// GetMPTentriesusedforEQ gets the value of MPTentriesusedforEQ for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyMPTentriesusedforEQ() (value uint32, err error) {
	retValue, err := instance.GetProperty("MPTentriesusedforEQ")
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

// SetMPTentriesusedforMR sets the value of MPTentriesusedforMR for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyMPTentriesusedforMR(value uint32) (err error) {
	return instance.SetProperty("MPTentriesusedforMR", (value))
}

// GetMPTentriesusedforMR gets the value of MPTentriesusedforMR for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyMPTentriesusedforMR() (value uint32, err error) {
	retValue, err := instance.GetProperty("MPTentriesusedforMR")
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

// SetMPTentriesusedforQP sets the value of MPTentriesusedforQP for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyMPTentriesusedforQP(value uint32) (err error) {
	return instance.SetProperty("MPTentriesusedforQP", (value))
}

// GetMPTentriesusedforQP gets the value of MPTentriesusedforQP for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyMPTentriesusedforQP() (value uint32, err error) {
	retValue, err := instance.GetProperty("MPTentriesusedforQP")
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

// SetMPTMissPersec sets the value of MPTMissPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyMPTMissPersec(value uint32) (err error) {
	return instance.SetProperty("MPTMissPersec", (value))
}

// GetMPTMissPersec gets the value of MPTMissPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyMPTMissPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("MPTMissPersec")
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

// SetMTTentriesusedforCQ sets the value of MTTentriesusedforCQ for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyMTTentriesusedforCQ(value uint32) (err error) {
	return instance.SetProperty("MTTentriesusedforCQ", (value))
}

// GetMTTentriesusedforCQ gets the value of MTTentriesusedforCQ for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyMTTentriesusedforCQ() (value uint32, err error) {
	retValue, err := instance.GetProperty("MTTentriesusedforCQ")
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

// SetMTTentriesusedforEQ sets the value of MTTentriesusedforEQ for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyMTTentriesusedforEQ(value uint32) (err error) {
	return instance.SetProperty("MTTentriesusedforEQ", (value))
}

// GetMTTentriesusedforEQ gets the value of MTTentriesusedforEQ for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyMTTentriesusedforEQ() (value uint32, err error) {
	retValue, err := instance.GetProperty("MTTentriesusedforEQ")
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

// SetMTTentriesusedforMR sets the value of MTTentriesusedforMR for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyMTTentriesusedforMR(value uint32) (err error) {
	return instance.SetProperty("MTTentriesusedforMR", (value))
}

// GetMTTentriesusedforMR gets the value of MTTentriesusedforMR for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyMTTentriesusedforMR() (value uint32, err error) {
	retValue, err := instance.GetProperty("MTTentriesusedforMR")
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

// SetMTTentriesusedforQP sets the value of MTTentriesusedforQP for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyMTTentriesusedforQP(value uint32) (err error) {
	return instance.SetProperty("MTTentriesusedforQP", (value))
}

// GetMTTentriesusedforQP gets the value of MTTentriesusedforQP for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyMTTentriesusedforQP() (value uint32, err error) {
	retValue, err := instance.GetProperty("MTTentriesusedforQP")
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

// SetMTTMissPersec sets the value of MTTMissPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyMTTMissPersec(value uint32) (err error) {
	return instance.SetProperty("MTTMissPersec", (value))
}

// GetMTTMissPersec gets the value of MTTMissPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyMTTMissPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("MTTMissPersec")
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

// SetNoWQEDropsPersec sets the value of NoWQEDropsPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyNoWQEDropsPersec(value uint32) (err error) {
	return instance.SetProperty("NoWQEDropsPersec", (value))
}

// GetNoWQEDropsPersec gets the value of NoWQEDropsPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyNoWQEDropsPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("NoWQEDropsPersec")
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

// SetPacketsdiscardedduetoinvalidqp sets the value of Packetsdiscardedduetoinvalidqp for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyPacketsdiscardedduetoinvalidqp(value uint64) (err error) {
	return instance.SetProperty("Packetsdiscardedduetoinvalidqp", (value))
}

// GetPacketsdiscardedduetoinvalidqp gets the value of Packetsdiscardedduetoinvalidqp for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyPacketsdiscardedduetoinvalidqp() (value uint64, err error) {
	retValue, err := instance.GetProperty("Packetsdiscardedduetoinvalidqp")
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

// SetPCIBackpressurePersec sets the value of PCIBackpressurePersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyPCIBackpressurePersec(value uint32) (err error) {
	return instance.SetProperty("PCIBackpressurePersec", (value))
}

// GetPCIBackpressurePersec gets the value of PCIBackpressurePersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyPCIBackpressurePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("PCIBackpressurePersec")
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

// SetQppriorityupdateflowevents sets the value of Qppriorityupdateflowevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyQppriorityupdateflowevents(value uint32) (err error) {
	return instance.SetProperty("Qppriorityupdateflowevents", (value))
}

// GetQppriorityupdateflowevents gets the value of Qppriorityupdateflowevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyQppriorityupdateflowevents() (value uint32, err error) {
	retValue, err := instance.GetProperty("Qppriorityupdateflowevents")
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

// SetReceiveWQEcachehitPersec sets the value of ReceiveWQEcachehitPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyReceiveWQEcachehitPersec(value uint32) (err error) {
	return instance.SetProperty("ReceiveWQEcachehitPersec", (value))
}

// GetReceiveWQEcachehitPersec gets the value of ReceiveWQEcachehitPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyReceiveWQEcachehitPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceiveWQEcachehitPersec")
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

// SetReceiveWQEcachelookupPersec sets the value of ReceiveWQEcachelookupPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyReceiveWQEcachelookupPersec(value uint32) (err error) {
	return instance.SetProperty("ReceiveWQEcachelookupPersec", (value))
}

// GetReceiveWQEcachelookupPersec gets the value of ReceiveWQEcachelookupPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyReceiveWQEcachelookupPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ReceiveWQEcachelookupPersec")
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

// SetRQMissPersec sets the value of RQMissPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyRQMissPersec(value uint32) (err error) {
	return instance.SetProperty("RQMissPersec", (value))
}

// GetRQMissPersec gets the value of RQMissPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyRQMissPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("RQMissPersec")
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

// SetScatterBackpressurePersec sets the value of ScatterBackpressurePersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyScatterBackpressurePersec(value uint32) (err error) {
	return instance.SetProperty("ScatterBackpressurePersec", (value))
}

// GetScatterBackpressurePersec gets the value of ScatterBackpressurePersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyScatterBackpressurePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("ScatterBackpressurePersec")
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

// SetSQMissPersec sets the value of SQMissPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertySQMissPersec(value uint32) (err error) {
	return instance.SetProperty("SQMissPersec", (value))
}

// GetSQMissPersec gets the value of SQMissPersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertySQMissPersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SQMissPersec")
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

// SetSteeringQPCBackpressurePersec sets the value of SteeringQPCBackpressurePersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertySteeringQPCBackpressurePersec(value uint32) (err error) {
	return instance.SetProperty("SteeringQPCBackpressurePersec", (value))
}

// GetSteeringQPCBackpressurePersec gets the value of SteeringQPCBackpressurePersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertySteeringQPCBackpressurePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("SteeringQPCBackpressurePersec")
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

// SetTotalqpsinlimitedstate sets the value of Totalqpsinlimitedstate for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyTotalqpsinlimitedstate(value uint32) (err error) {
	return instance.SetProperty("Totalqpsinlimitedstate", (value))
}

// GetTotalqpsinlimitedstate gets the value of Totalqpsinlimitedstate for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyTotalqpsinlimitedstate() (value uint32, err error) {
	retValue, err := instance.GetProperty("Totalqpsinlimitedstate")
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

// SetTransmissionenginehangevents sets the value of Transmissionenginehangevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyTransmissionenginehangevents(value uint32) (err error) {
	return instance.SetProperty("Transmissionenginehangevents", (value))
}

// GetTransmissionenginehangevents gets the value of Transmissionenginehangevents for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyTransmissionenginehangevents() (value uint32, err error) {
	retValue, err := instance.GetProperty("Transmissionenginehangevents")
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

// SetWQEfetchPerAtomicBackpressurePersec sets the value of WQEfetchPerAtomicBackpressurePersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) SetPropertyWQEfetchPerAtomicBackpressurePersec(value uint32) (err error) {
	return instance.SetProperty("WQEfetchPerAtomicBackpressurePersec", (value))
}

// GetWQEfetchPerAtomicBackpressurePersec gets the value of WQEfetchPerAtomicBackpressurePersec for the instance
func (instance *Win32_PerfRawData_Mlx4DeviceProvider_MellanoxWinOFBusCounters) GetPropertyWQEfetchPerAtomicBackpressurePersec() (value uint32, err error) {
	retValue, err := instance.GetProperty("WQEfetchPerAtomicBackpressurePersec")
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

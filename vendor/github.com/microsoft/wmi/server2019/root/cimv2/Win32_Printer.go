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

// Win32_Printer struct
type Win32_Printer struct {
	*CIM_Printer

	// The Attributes property indicates the attributes of the Win32 printing device. These attributes are represented through a combination of flags. Attributes of the printer include:
	///Queued  - Print jobs are buffered and queued.
	/// Direct  - Specifies that the document should be sent directly to the printer.  This is used if print job are not being properly queued.
	///Default - The printer is the default printer on the computer.
	///Shared - Available as a shared network resource.
	///Network - Attached to the network.
	///Hidden - Hidden from some users on the network.
	///Local - Directly connected to this computer.
	///EnableDevQ - Enable the queue on the printer if available.
	///KeepPrintedJobs - Specifies that the spooler should not delete documents after they are printed.
	///DoCompleteFirst - Start jobs that are finished spooling first.
	///WorkOffline - Queue print jobs when printer is not available.
	///EnableBIDI - Enable bi-directional printing.
	///RawOnly - Allow only raw data type jobs to be spooled.
	///Published - Indicates whether the printer is published in the network directory service.
	///
	Attributes uint32

	// The AveragePagesPerMinute property specifies the rate (average number of pages per minute) that the printer is capable of sustaining.
	AveragePagesPerMinute uint32

	// The Comment property specifies the comment of a print queue.
	///Example: Color printer
	Comment string

	// The Default property indicates whether the printer is the default printer on the computer.
	Default bool

	// The DefaultPriority property specifies the default priority value assigned to each print job.
	DefaultPriority uint32

	// The Direct property indicates whether the print jobs should be sent directly to the printer.  This means that no spool files are created for the print jobs.
	///
	Direct bool

	// The DoCompleteFirst property indicates whether the printer should start jobs that have finished spooling as opposed to the order of the job received.
	DoCompleteFirst bool

	// The DriverName property specifies the name of the Win32 printer driver.
	///Example: Windows NT Fax Driver
	DriverName string

	// The EnableBIDI property indicates whether the printer can print bidirectionally.
	EnableBIDI bool

	// The EnableDevQueryPrint property indicates whether to hold documents in the queue, if document and printer setups do not match
	EnableDevQueryPrint bool

	// The ExtendedDetectedErrorState property reports standard error information.  Any additional information should be recorded in the DetecteErrorState property.
	ExtendedDetectedErrorState Printer_ExtendedDetectedErrorState

	// Status information for a Printer, beyond that specified in the LogicalDevice Availability property. Values include "Idle" (3) and an indication that the Device is currently printing (4).
	ExtendedPrinterStatus Printer_ExtendedPrinterStatus

	// The Hidden property indicates whether the printer is hidden from network users.
	Hidden bool

	// The KeepPrintedJobs property indicates whether the print spooler should not delete the jobs after they are completed.
	KeepPrintedJobs bool

	// The Local property indicates whether the printer is attached to the network.  A masquerading printer is printer that is implemented as local printers but has a port that refers to a remote machine.  From the application perspective these hybrid printers should be viewed as printer connections since that is their intended behavior.
	Local bool

	// The Location property specifies the physical location of the printer.
	///Example: Bldg. 38, Room 1164
	Location string

	// The Network property indicates whether the printer is a network printer.
	Network bool

	// The Parameters property specifies optional parameters for the print processor.
	///Example: Copies=2
	Parameters string

	// The PortName property identifies the ports that can be used to transmit data to the printer. If a printer is connected to more than one port, the names of each port are separated by commas. Under Windows 95, only one port can be specified.
	///Example: LPT1:, LPT2:, LPT3:
	PortName string

	// The PrinterPaperNames property indicates the list of paper sizes supported by the printer. The printer-specified names are used to represent supported paper sizes.
	///Example: B5 (JIS).
	PrinterPaperNames []string

	// This property has been deprecated in favor of PrinterStatus, DetectedErrorState and ErrorInformation CIM properties that more clearly indicate the state and error status of the printer. The PrinterState property specifies a values indicating one of the possible states relating to this printer.
	PrinterState Printer_PrinterState

	// The PrintJobDataType property indicates the default data type that will be used for a print job.
	PrintJobDataType string

	// The PrintProcessor property specifies the name of the print spooler that handles print jobs.
	///Example: SPOOLSS.DLL.
	PrintProcessor string

	// The Priority property specifies the priority of the  printer. The jobs on a higher priority printer are scheduled first.
	Priority uint32

	// The Published property indicates whether the printer is published in the network directory service.
	Published bool

	// The Queued property indicates whether the printer buffers and queues print jobs.
	Queued bool

	// The RawOnly property indicates whether the printer accepts only raw data to be spooled.
	RawOnly bool

	// The SeparatorFile property specifies the name of the file used to create a separator page. This page is used to separate print jobs sent to the printer.
	SeparatorFile string

	// The ServerName property identifies the server that controls the printer. If this string is NULL, the printer is controlled locally.
	ServerName string

	// The Shared property indicates whether the printer is available as a shared network resource.
	Shared bool

	// The ShareName property indicates the share name of the Win32 printing device.
	///Example: \\PRINTSERVER1\PRINTER2
	ShareName string

	// The SpoolEnabled property shows whether spooling is enabled for this printer.
	///Values:TRUE or FALSE.
	///The SpoolEnabled property has been deprecated.  There is no replacementvalue and this property is now considered obsolete.
	SpoolEnabled bool

	// The StartTime property specifies the earliest time the printer can print a job (if the printer has been limited to print only at certain times). This value is expressed as time elapsed since 12:00 AM GMT (Greenwich mean time).
	StartTime string

	// The UntilTime property specifies the latest time the printer can print a job (if the printer has been limited to print only at certain times). This value is expressed as time elapsed since 12:00 AM GMT (Greenwich mean time).
	UntilTime string

	// The WorkOffline property indicates whether to queue print jobs on the computer if the printer is offline.
	WorkOffline bool
}

func NewWin32_PrinterEx1(instance *cim.WmiInstance) (newInstance *Win32_Printer, err error) {
	tmp, err := NewCIM_PrinterEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_Printer{
		CIM_Printer: tmp,
	}
	return
}

func NewWin32_PrinterEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_Printer, err error) {
	tmp, err := NewCIM_PrinterEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_Printer{
		CIM_Printer: tmp,
	}
	return
}

// SetAttributes sets the value of Attributes for the instance
func (instance *Win32_Printer) SetPropertyAttributes(value uint32) (err error) {
	return instance.SetProperty("Attributes", (value))
}

// GetAttributes gets the value of Attributes for the instance
func (instance *Win32_Printer) GetPropertyAttributes() (value uint32, err error) {
	retValue, err := instance.GetProperty("Attributes")
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

// SetAveragePagesPerMinute sets the value of AveragePagesPerMinute for the instance
func (instance *Win32_Printer) SetPropertyAveragePagesPerMinute(value uint32) (err error) {
	return instance.SetProperty("AveragePagesPerMinute", (value))
}

// GetAveragePagesPerMinute gets the value of AveragePagesPerMinute for the instance
func (instance *Win32_Printer) GetPropertyAveragePagesPerMinute() (value uint32, err error) {
	retValue, err := instance.GetProperty("AveragePagesPerMinute")
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

// SetComment sets the value of Comment for the instance
func (instance *Win32_Printer) SetPropertyComment(value string) (err error) {
	return instance.SetProperty("Comment", (value))
}

// GetComment gets the value of Comment for the instance
func (instance *Win32_Printer) GetPropertyComment() (value string, err error) {
	retValue, err := instance.GetProperty("Comment")
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

// SetDefault sets the value of Default for the instance
func (instance *Win32_Printer) SetPropertyDefault(value bool) (err error) {
	return instance.SetProperty("Default", (value))
}

// GetDefault gets the value of Default for the instance
func (instance *Win32_Printer) GetPropertyDefault() (value bool, err error) {
	retValue, err := instance.GetProperty("Default")
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

// SetDefaultPriority sets the value of DefaultPriority for the instance
func (instance *Win32_Printer) SetPropertyDefaultPriority(value uint32) (err error) {
	return instance.SetProperty("DefaultPriority", (value))
}

// GetDefaultPriority gets the value of DefaultPriority for the instance
func (instance *Win32_Printer) GetPropertyDefaultPriority() (value uint32, err error) {
	retValue, err := instance.GetProperty("DefaultPriority")
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

// SetDirect sets the value of Direct for the instance
func (instance *Win32_Printer) SetPropertyDirect(value bool) (err error) {
	return instance.SetProperty("Direct", (value))
}

// GetDirect gets the value of Direct for the instance
func (instance *Win32_Printer) GetPropertyDirect() (value bool, err error) {
	retValue, err := instance.GetProperty("Direct")
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

// SetDoCompleteFirst sets the value of DoCompleteFirst for the instance
func (instance *Win32_Printer) SetPropertyDoCompleteFirst(value bool) (err error) {
	return instance.SetProperty("DoCompleteFirst", (value))
}

// GetDoCompleteFirst gets the value of DoCompleteFirst for the instance
func (instance *Win32_Printer) GetPropertyDoCompleteFirst() (value bool, err error) {
	retValue, err := instance.GetProperty("DoCompleteFirst")
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

// SetDriverName sets the value of DriverName for the instance
func (instance *Win32_Printer) SetPropertyDriverName(value string) (err error) {
	return instance.SetProperty("DriverName", (value))
}

// GetDriverName gets the value of DriverName for the instance
func (instance *Win32_Printer) GetPropertyDriverName() (value string, err error) {
	retValue, err := instance.GetProperty("DriverName")
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

// SetEnableBIDI sets the value of EnableBIDI for the instance
func (instance *Win32_Printer) SetPropertyEnableBIDI(value bool) (err error) {
	return instance.SetProperty("EnableBIDI", (value))
}

// GetEnableBIDI gets the value of EnableBIDI for the instance
func (instance *Win32_Printer) GetPropertyEnableBIDI() (value bool, err error) {
	retValue, err := instance.GetProperty("EnableBIDI")
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

// SetEnableDevQueryPrint sets the value of EnableDevQueryPrint for the instance
func (instance *Win32_Printer) SetPropertyEnableDevQueryPrint(value bool) (err error) {
	return instance.SetProperty("EnableDevQueryPrint", (value))
}

// GetEnableDevQueryPrint gets the value of EnableDevQueryPrint for the instance
func (instance *Win32_Printer) GetPropertyEnableDevQueryPrint() (value bool, err error) {
	retValue, err := instance.GetProperty("EnableDevQueryPrint")
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

// SetExtendedDetectedErrorState sets the value of ExtendedDetectedErrorState for the instance
func (instance *Win32_Printer) SetPropertyExtendedDetectedErrorState(value Printer_ExtendedDetectedErrorState) (err error) {
	return instance.SetProperty("ExtendedDetectedErrorState", (value))
}

// GetExtendedDetectedErrorState gets the value of ExtendedDetectedErrorState for the instance
func (instance *Win32_Printer) GetPropertyExtendedDetectedErrorState() (value Printer_ExtendedDetectedErrorState, err error) {
	retValue, err := instance.GetProperty("ExtendedDetectedErrorState")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Printer_ExtendedDetectedErrorState(valuetmp)

	return
}

// SetExtendedPrinterStatus sets the value of ExtendedPrinterStatus for the instance
func (instance *Win32_Printer) SetPropertyExtendedPrinterStatus(value Printer_ExtendedPrinterStatus) (err error) {
	return instance.SetProperty("ExtendedPrinterStatus", (value))
}

// GetExtendedPrinterStatus gets the value of ExtendedPrinterStatus for the instance
func (instance *Win32_Printer) GetPropertyExtendedPrinterStatus() (value Printer_ExtendedPrinterStatus, err error) {
	retValue, err := instance.GetProperty("ExtendedPrinterStatus")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Printer_ExtendedPrinterStatus(valuetmp)

	return
}

// SetHidden sets the value of Hidden for the instance
func (instance *Win32_Printer) SetPropertyHidden(value bool) (err error) {
	return instance.SetProperty("Hidden", (value))
}

// GetHidden gets the value of Hidden for the instance
func (instance *Win32_Printer) GetPropertyHidden() (value bool, err error) {
	retValue, err := instance.GetProperty("Hidden")
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

// SetKeepPrintedJobs sets the value of KeepPrintedJobs for the instance
func (instance *Win32_Printer) SetPropertyKeepPrintedJobs(value bool) (err error) {
	return instance.SetProperty("KeepPrintedJobs", (value))
}

// GetKeepPrintedJobs gets the value of KeepPrintedJobs for the instance
func (instance *Win32_Printer) GetPropertyKeepPrintedJobs() (value bool, err error) {
	retValue, err := instance.GetProperty("KeepPrintedJobs")
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

// SetLocal sets the value of Local for the instance
func (instance *Win32_Printer) SetPropertyLocal(value bool) (err error) {
	return instance.SetProperty("Local", (value))
}

// GetLocal gets the value of Local for the instance
func (instance *Win32_Printer) GetPropertyLocal() (value bool, err error) {
	retValue, err := instance.GetProperty("Local")
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

// SetLocation sets the value of Location for the instance
func (instance *Win32_Printer) SetPropertyLocation(value string) (err error) {
	return instance.SetProperty("Location", (value))
}

// GetLocation gets the value of Location for the instance
func (instance *Win32_Printer) GetPropertyLocation() (value string, err error) {
	retValue, err := instance.GetProperty("Location")
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

// SetNetwork sets the value of Network for the instance
func (instance *Win32_Printer) SetPropertyNetwork(value bool) (err error) {
	return instance.SetProperty("Network", (value))
}

// GetNetwork gets the value of Network for the instance
func (instance *Win32_Printer) GetPropertyNetwork() (value bool, err error) {
	retValue, err := instance.GetProperty("Network")
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

// SetParameters sets the value of Parameters for the instance
func (instance *Win32_Printer) SetPropertyParameters(value string) (err error) {
	return instance.SetProperty("Parameters", (value))
}

// GetParameters gets the value of Parameters for the instance
func (instance *Win32_Printer) GetPropertyParameters() (value string, err error) {
	retValue, err := instance.GetProperty("Parameters")
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

// SetPortName sets the value of PortName for the instance
func (instance *Win32_Printer) SetPropertyPortName(value string) (err error) {
	return instance.SetProperty("PortName", (value))
}

// GetPortName gets the value of PortName for the instance
func (instance *Win32_Printer) GetPropertyPortName() (value string, err error) {
	retValue, err := instance.GetProperty("PortName")
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

// SetPrinterPaperNames sets the value of PrinterPaperNames for the instance
func (instance *Win32_Printer) SetPropertyPrinterPaperNames(value []string) (err error) {
	return instance.SetProperty("PrinterPaperNames", (value))
}

// GetPrinterPaperNames gets the value of PrinterPaperNames for the instance
func (instance *Win32_Printer) GetPropertyPrinterPaperNames() (value []string, err error) {
	retValue, err := instance.GetProperty("PrinterPaperNames")
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

// SetPrinterState sets the value of PrinterState for the instance
func (instance *Win32_Printer) SetPropertyPrinterState(value Printer_PrinterState) (err error) {
	return instance.SetProperty("PrinterState", (value))
}

// GetPrinterState gets the value of PrinterState for the instance
func (instance *Win32_Printer) GetPropertyPrinterState() (value Printer_PrinterState, err error) {
	retValue, err := instance.GetProperty("PrinterState")
	if err != nil {
		return
	}
	if retValue == nil {
		// Doesn't have any value. Return empty
		return
	}

	valuetmp, ok := retValue.(int32)
	if !ok {
		err = errors.Wrapf(errors.InvalidType, " int32 is Invalid. Expected %s", reflect.TypeOf(retValue))
		return
	}

	value = Printer_PrinterState(valuetmp)

	return
}

// SetPrintJobDataType sets the value of PrintJobDataType for the instance
func (instance *Win32_Printer) SetPropertyPrintJobDataType(value string) (err error) {
	return instance.SetProperty("PrintJobDataType", (value))
}

// GetPrintJobDataType gets the value of PrintJobDataType for the instance
func (instance *Win32_Printer) GetPropertyPrintJobDataType() (value string, err error) {
	retValue, err := instance.GetProperty("PrintJobDataType")
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

// SetPrintProcessor sets the value of PrintProcessor for the instance
func (instance *Win32_Printer) SetPropertyPrintProcessor(value string) (err error) {
	return instance.SetProperty("PrintProcessor", (value))
}

// GetPrintProcessor gets the value of PrintProcessor for the instance
func (instance *Win32_Printer) GetPropertyPrintProcessor() (value string, err error) {
	retValue, err := instance.GetProperty("PrintProcessor")
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

// SetPriority sets the value of Priority for the instance
func (instance *Win32_Printer) SetPropertyPriority(value uint32) (err error) {
	return instance.SetProperty("Priority", (value))
}

// GetPriority gets the value of Priority for the instance
func (instance *Win32_Printer) GetPropertyPriority() (value uint32, err error) {
	retValue, err := instance.GetProperty("Priority")
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

// SetPublished sets the value of Published for the instance
func (instance *Win32_Printer) SetPropertyPublished(value bool) (err error) {
	return instance.SetProperty("Published", (value))
}

// GetPublished gets the value of Published for the instance
func (instance *Win32_Printer) GetPropertyPublished() (value bool, err error) {
	retValue, err := instance.GetProperty("Published")
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

// SetQueued sets the value of Queued for the instance
func (instance *Win32_Printer) SetPropertyQueued(value bool) (err error) {
	return instance.SetProperty("Queued", (value))
}

// GetQueued gets the value of Queued for the instance
func (instance *Win32_Printer) GetPropertyQueued() (value bool, err error) {
	retValue, err := instance.GetProperty("Queued")
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

// SetRawOnly sets the value of RawOnly for the instance
func (instance *Win32_Printer) SetPropertyRawOnly(value bool) (err error) {
	return instance.SetProperty("RawOnly", (value))
}

// GetRawOnly gets the value of RawOnly for the instance
func (instance *Win32_Printer) GetPropertyRawOnly() (value bool, err error) {
	retValue, err := instance.GetProperty("RawOnly")
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

// SetSeparatorFile sets the value of SeparatorFile for the instance
func (instance *Win32_Printer) SetPropertySeparatorFile(value string) (err error) {
	return instance.SetProperty("SeparatorFile", (value))
}

// GetSeparatorFile gets the value of SeparatorFile for the instance
func (instance *Win32_Printer) GetPropertySeparatorFile() (value string, err error) {
	retValue, err := instance.GetProperty("SeparatorFile")
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

// SetServerName sets the value of ServerName for the instance
func (instance *Win32_Printer) SetPropertyServerName(value string) (err error) {
	return instance.SetProperty("ServerName", (value))
}

// GetServerName gets the value of ServerName for the instance
func (instance *Win32_Printer) GetPropertyServerName() (value string, err error) {
	retValue, err := instance.GetProperty("ServerName")
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

// SetShared sets the value of Shared for the instance
func (instance *Win32_Printer) SetPropertyShared(value bool) (err error) {
	return instance.SetProperty("Shared", (value))
}

// GetShared gets the value of Shared for the instance
func (instance *Win32_Printer) GetPropertyShared() (value bool, err error) {
	retValue, err := instance.GetProperty("Shared")
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

// SetShareName sets the value of ShareName for the instance
func (instance *Win32_Printer) SetPropertyShareName(value string) (err error) {
	return instance.SetProperty("ShareName", (value))
}

// GetShareName gets the value of ShareName for the instance
func (instance *Win32_Printer) GetPropertyShareName() (value string, err error) {
	retValue, err := instance.GetProperty("ShareName")
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

// SetSpoolEnabled sets the value of SpoolEnabled for the instance
func (instance *Win32_Printer) SetPropertySpoolEnabled(value bool) (err error) {
	return instance.SetProperty("SpoolEnabled", (value))
}

// GetSpoolEnabled gets the value of SpoolEnabled for the instance
func (instance *Win32_Printer) GetPropertySpoolEnabled() (value bool, err error) {
	retValue, err := instance.GetProperty("SpoolEnabled")
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

// SetStartTime sets the value of StartTime for the instance
func (instance *Win32_Printer) SetPropertyStartTime(value string) (err error) {
	return instance.SetProperty("StartTime", (value))
}

// GetStartTime gets the value of StartTime for the instance
func (instance *Win32_Printer) GetPropertyStartTime() (value string, err error) {
	retValue, err := instance.GetProperty("StartTime")
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

// SetUntilTime sets the value of UntilTime for the instance
func (instance *Win32_Printer) SetPropertyUntilTime(value string) (err error) {
	return instance.SetProperty("UntilTime", (value))
}

// GetUntilTime gets the value of UntilTime for the instance
func (instance *Win32_Printer) GetPropertyUntilTime() (value string, err error) {
	retValue, err := instance.GetProperty("UntilTime")
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

// SetWorkOffline sets the value of WorkOffline for the instance
func (instance *Win32_Printer) SetPropertyWorkOffline(value bool) (err error) {
	return instance.SetProperty("WorkOffline", (value))
}

// GetWorkOffline gets the value of WorkOffline for the instance
func (instance *Win32_Printer) GetPropertyWorkOffline() (value bool, err error) {
	retValue, err := instance.GetProperty("WorkOffline")
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

// The Pause method pauses the print queue. No jobs can print anymore until the print queue is resumed. The method can return the following values:
///0 - Success.
///5 - Access denied.
///Other - For integer values other than those listed above, refer to the documentation on the Win32 error codes.

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Printer) Pause() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Pause")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

// The Resume method resumes a paused print queue. The method can return the following values:
///0 - Success.
///5 - Access denied.
///Other - For integer values other than those listed above, refer to the documentation on the Win32 error codes.

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Printer) Resume() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Resume")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

// The CancelAllJobs method cancels and removes all print jobs from the printer queue including the job currently printing. The method can return the following values:
///0 - Success.
///5 - Access denied.
///Other - For integer values other than those listed above, refer to the documentation on the Win32 error codes.

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Printer) CancelAllJobs() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("CancelAllJobs")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

// The AddPrinterConnection method provides a connection to an existing printer on the network and adds it to the list of available printers on the computer system. If successful, applications will be able to use this printer for print jobs.  If unsuccessful the printer is not installed. The method can return the following values:
///0 - Success.
///5 - Access denied.
///1801 - Invalid printer name.
///1930 - Incompatible printer driver.
///Other - For integer values other than those listed above, refer to the documentation on the Win32 error codes.

// <param name="Name" type="string ">The Name parameter specifies a friendly name for the printer.  This may be overridden if the name has alreadybeen set by the printer.</param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Printer) AddPrinterConnection( /* IN */ Name string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("AddPrinterConnection", Name)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

// The RenamePrinter method renames a printer. The method can return the following values:
///0 - Success.
///5 - Access denied.
///1801 - Invalid printer name.
///Other - For integer values other than those listed above, refer to the documentation on the Win32 error codes.

// <param name="NewPrinterName" type="string ">The NewPrinterName parameter specifies the new printer name.</param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Printer) RenamePrinter( /* IN */ NewPrinterName string) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("RenamePrinter", NewPrinterName)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

// The PrintTestPage method prints a test page. The method can return the following values:
///0 - Success.
///5 - Access denied.
///Other - For integer values other than those listed above, refer to the documentation on the Win32 error codes.

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Printer) PrintTestPage() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("PrintTestPage")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

// The SetDefaultPrinter method sets the printer to be the default printer for the user who executes the method. The method can return the following values:
///0 - Success.
///Other - For integer values other than those listed above, refer to the documentation on the Win32 error codes.

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Printer) SetDefaultPrinter() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetDefaultPrinter")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

// Retrieves a structural representation of the printer's security descriptor.
///The method returns an integer value that can be interpreted as follows:
///0 - Successful completion.
///2 - The user does not have access to the requested information.
///8 - Unknown failure.
///9 - The user does not have adequate privileges.
///21 - The specified parameter is invalid.
///Other - For integer values other than those listed above, refer to Win32 error code documentation.

// <param name="Descriptor" type="Win32_SecurityDescriptor "></param>
// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Printer) GetSecurityDescriptor( /* OUT */ Descriptor Win32_SecurityDescriptor) (result uint32, err error) {
	retVal, err := instance.InvokeMethod("GetSecurityDescriptor")
	if err != nil {
		return
	}
	retValue := retVal[0].(int32)
	result = uint32(retValue)
	return

}

// Sets security descriptor on the printer to the specified structure.
///The method returns an integer value that can be interpreted as follows:
///0 - Successful completion.
///2 - The user does not have access to the requested information.
///8 - Unknown failure.
///9 - The user does not have adequate privileges.
///21 - The specified parameter is invalid.
///Other - For integer values other than those listed above, refer to Win32 error code documentation.

// <param name="Descriptor" type="Win32_SecurityDescriptor "></param>

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_Printer) SetSecurityDescriptor( /* IN */ Descriptor Win32_SecurityDescriptor) (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("SetSecurityDescriptor", Descriptor)
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

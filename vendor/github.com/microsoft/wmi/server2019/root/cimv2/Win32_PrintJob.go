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

// Win32_PrintJob struct
type Win32_PrintJob struct {
	*CIM_Job

	// The Color property indicates whether the document is to be printed in color or monochrome.  Some color printers have the capability to print using true black instead of a combination of Yellow, Cyan, and Magenta.  This usually creates darker and sharper text for documents.  This option is only useful for color printers that support true black printing.
	Color string

	// The DataType property indicates the format of the data for this print job. This instructs the printer driver to eithertranslate the data (generic text, PostScript, or PCL) before printing, or to print in a raw format (for graphics and pictures).
	///Example: TEXT
	DataType string

	// The Document property specifies the name of the print job. The user sees this name when viewing documents waiting to be printed.
	///Example: Microsoft Word - Review.doc
	Document string

	// The DriverName property indicates the name of the printer driver used for the print job.
	DriverName string

	// The HostPrintQueue property contains the name of the computer on which the print job was created.
	HostPrintQueue string

	// The JobId property indicates the identifier number of the job. It is used by other methods as a handle to a single job spooling to the printer.
	JobId uint32

	// The PagesPrinted property specifies the number of pages that have been printed. This value may be zero if the print job does not contain page delimiting information.
	PagesPrinted uint32

	// The PaperLength property indicates the length of the paper.
	///Example: 2794
	PaperLength uint32

	// The PaperSize property indicates the size of the paper.
	///Example: A4 or Letter
	PaperSize string

	// The PaperWidth property indicates the width of the paper.
	///Example: 2159
	PaperWidth uint32

	// The Parameters property indicates optional parameters to send to the print processor. See the PrintProcessor member for more information.
	Parameters string

	// The PrintProcessor property indicates the print processor service used to process the print job. A printer processor works in conjunction with the printer driver to provide additional translation of printer data for the printer, and can also be used to provide special options such as a title page for the job.
	PrintProcessor string

	// The Size property indicates the size of the print job.
	Size uint32

	// The SizeHigh property indicates the size of the print job if the Size property exceeds 4,294,967,295 bytes.
	SizeHigh uint32

	// The StatusMask property specifies a bitmap of the possible statuses relating to this print job.
	StatusMask uint32

	// The TotalPages property specifies the number of pages required to complete the job. This value may be zero if the print job does not contain page-delimiting information.
	TotalPages uint32
}

func NewWin32_PrintJobEx1(instance *cim.WmiInstance) (newInstance *Win32_PrintJob, err error) {
	tmp, err := NewCIM_JobEx1(instance)

	if err != nil {
		return
	}
	newInstance = &Win32_PrintJob{
		CIM_Job: tmp,
	}
	return
}

func NewWin32_PrintJobEx6(hostName string,
	wmiNamespace string,
	userName string,
	password string,
	domainName string,
	query *query.WmiQuery) (newInstance *Win32_PrintJob, err error) {
	tmp, err := NewCIM_JobEx6(hostName, wmiNamespace, userName, password, domainName, query)

	if err != nil {
		return
	}
	newInstance = &Win32_PrintJob{
		CIM_Job: tmp,
	}
	return
}

// SetColor sets the value of Color for the instance
func (instance *Win32_PrintJob) SetPropertyColor(value string) (err error) {
	return instance.SetProperty("Color", (value))
}

// GetColor gets the value of Color for the instance
func (instance *Win32_PrintJob) GetPropertyColor() (value string, err error) {
	retValue, err := instance.GetProperty("Color")
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

// SetDataType sets the value of DataType for the instance
func (instance *Win32_PrintJob) SetPropertyDataType(value string) (err error) {
	return instance.SetProperty("DataType", (value))
}

// GetDataType gets the value of DataType for the instance
func (instance *Win32_PrintJob) GetPropertyDataType() (value string, err error) {
	retValue, err := instance.GetProperty("DataType")
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

// SetDocument sets the value of Document for the instance
func (instance *Win32_PrintJob) SetPropertyDocument(value string) (err error) {
	return instance.SetProperty("Document", (value))
}

// GetDocument gets the value of Document for the instance
func (instance *Win32_PrintJob) GetPropertyDocument() (value string, err error) {
	retValue, err := instance.GetProperty("Document")
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

// SetDriverName sets the value of DriverName for the instance
func (instance *Win32_PrintJob) SetPropertyDriverName(value string) (err error) {
	return instance.SetProperty("DriverName", (value))
}

// GetDriverName gets the value of DriverName for the instance
func (instance *Win32_PrintJob) GetPropertyDriverName() (value string, err error) {
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

// SetHostPrintQueue sets the value of HostPrintQueue for the instance
func (instance *Win32_PrintJob) SetPropertyHostPrintQueue(value string) (err error) {
	return instance.SetProperty("HostPrintQueue", (value))
}

// GetHostPrintQueue gets the value of HostPrintQueue for the instance
func (instance *Win32_PrintJob) GetPropertyHostPrintQueue() (value string, err error) {
	retValue, err := instance.GetProperty("HostPrintQueue")
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

// SetJobId sets the value of JobId for the instance
func (instance *Win32_PrintJob) SetPropertyJobId(value uint32) (err error) {
	return instance.SetProperty("JobId", (value))
}

// GetJobId gets the value of JobId for the instance
func (instance *Win32_PrintJob) GetPropertyJobId() (value uint32, err error) {
	retValue, err := instance.GetProperty("JobId")
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

// SetPagesPrinted sets the value of PagesPrinted for the instance
func (instance *Win32_PrintJob) SetPropertyPagesPrinted(value uint32) (err error) {
	return instance.SetProperty("PagesPrinted", (value))
}

// GetPagesPrinted gets the value of PagesPrinted for the instance
func (instance *Win32_PrintJob) GetPropertyPagesPrinted() (value uint32, err error) {
	retValue, err := instance.GetProperty("PagesPrinted")
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

// SetPaperLength sets the value of PaperLength for the instance
func (instance *Win32_PrintJob) SetPropertyPaperLength(value uint32) (err error) {
	return instance.SetProperty("PaperLength", (value))
}

// GetPaperLength gets the value of PaperLength for the instance
func (instance *Win32_PrintJob) GetPropertyPaperLength() (value uint32, err error) {
	retValue, err := instance.GetProperty("PaperLength")
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

// SetPaperSize sets the value of PaperSize for the instance
func (instance *Win32_PrintJob) SetPropertyPaperSize(value string) (err error) {
	return instance.SetProperty("PaperSize", (value))
}

// GetPaperSize gets the value of PaperSize for the instance
func (instance *Win32_PrintJob) GetPropertyPaperSize() (value string, err error) {
	retValue, err := instance.GetProperty("PaperSize")
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

// SetPaperWidth sets the value of PaperWidth for the instance
func (instance *Win32_PrintJob) SetPropertyPaperWidth(value uint32) (err error) {
	return instance.SetProperty("PaperWidth", (value))
}

// GetPaperWidth gets the value of PaperWidth for the instance
func (instance *Win32_PrintJob) GetPropertyPaperWidth() (value uint32, err error) {
	retValue, err := instance.GetProperty("PaperWidth")
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

// SetParameters sets the value of Parameters for the instance
func (instance *Win32_PrintJob) SetPropertyParameters(value string) (err error) {
	return instance.SetProperty("Parameters", (value))
}

// GetParameters gets the value of Parameters for the instance
func (instance *Win32_PrintJob) GetPropertyParameters() (value string, err error) {
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

// SetPrintProcessor sets the value of PrintProcessor for the instance
func (instance *Win32_PrintJob) SetPropertyPrintProcessor(value string) (err error) {
	return instance.SetProperty("PrintProcessor", (value))
}

// GetPrintProcessor gets the value of PrintProcessor for the instance
func (instance *Win32_PrintJob) GetPropertyPrintProcessor() (value string, err error) {
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

// SetSize sets the value of Size for the instance
func (instance *Win32_PrintJob) SetPropertySize(value uint32) (err error) {
	return instance.SetProperty("Size", (value))
}

// GetSize gets the value of Size for the instance
func (instance *Win32_PrintJob) GetPropertySize() (value uint32, err error) {
	retValue, err := instance.GetProperty("Size")
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

// SetSizeHigh sets the value of SizeHigh for the instance
func (instance *Win32_PrintJob) SetPropertySizeHigh(value uint32) (err error) {
	return instance.SetProperty("SizeHigh", (value))
}

// GetSizeHigh gets the value of SizeHigh for the instance
func (instance *Win32_PrintJob) GetPropertySizeHigh() (value uint32, err error) {
	retValue, err := instance.GetProperty("SizeHigh")
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

// SetStatusMask sets the value of StatusMask for the instance
func (instance *Win32_PrintJob) SetPropertyStatusMask(value uint32) (err error) {
	return instance.SetProperty("StatusMask", (value))
}

// GetStatusMask gets the value of StatusMask for the instance
func (instance *Win32_PrintJob) GetPropertyStatusMask() (value uint32, err error) {
	retValue, err := instance.GetProperty("StatusMask")
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

// SetTotalPages sets the value of TotalPages for the instance
func (instance *Win32_PrintJob) SetPropertyTotalPages(value uint32) (err error) {
	return instance.SetProperty("TotalPages", (value))
}

// GetTotalPages gets the value of TotalPages for the instance
func (instance *Win32_PrintJob) GetPropertyTotalPages() (value uint32, err error) {
	retValue, err := instance.GetProperty("TotalPages")
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

// The Pause method pauses a job in a print queue. If the job was currently printing, no other job will be printed. If the job wasn't printing yet, another unpaused print job may begin printing. The method can return the following values:
///0 - Success.
///5 - Access denied.
///Other - For integer values other than those listed above, refer to the documentation on the Win32 error codes.

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_PrintJob) Pause() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Pause")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

// The Resume method continues a paused print job. The method can return the following values:
///0 - Success.
///5 - Access denied.
///Other - For integer values other than those listed above, refer to the documentation on the Win32 error codes.

// <param name="ReturnValue" type="uint32 "></param>
func (instance *Win32_PrintJob) Resume() (result uint32, err error) {
	retVal, err := instance.InvokeMethodWithReturn("Resume")
	if err != nil {
		return
	}
	result = uint32(retVal)
	return

}

//go:build windows
// +build windows

package cim

import (
	"fmt"

	"github.com/microsoft/wmi/pkg/base/query"
	"github.com/microsoft/wmi/server2019/root/cimv2"
)

var (
	BIOSSelectorList    = []string{"SerialNumber"}
	ServiceSelectorList = []string{"DisplayName", "State", "StartMode"}
)

type ServiceInterface interface {
	GetPropertyName() (string, error)
	GetPropertyDisplayName() (string, error)
	GetPropertyState() (string, error)
	GetPropertyStartMode() (string, error)
	GetDependents() ([]ServiceInterface, error)
	StartService() (result uint32, err error)
	StopService() (result uint32, err error)
	Refresh() error
}

// QueryBIOSElement retrieves the BIOS element.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM CIM_BIOSElement
//
// Refer to https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/cim-bioselement
// for the WMI class definition.
func QueryBIOSElement(selectorList []string) (*cimv2.CIM_BIOSElement, error) {
	biosQuery := query.NewWmiQueryWithSelectList("CIM_BIOSElement", selectorList)
	instances, err := QueryInstances("", biosQuery)
	if err != nil {
		return nil, err
	}

	bios, err := cimv2.NewCIM_BIOSElementEx1(instances[0])
	if err != nil {
		return nil, fmt.Errorf("failed to get BIOS element: %w", err)
	}

	return bios, err
}

// GetBIOSSerialNumber returns the BIOS serial number.
func GetBIOSSerialNumber(bios *cimv2.CIM_BIOSElement) (string, error) {
	return bios.GetPropertySerialNumber()
}

// QueryServiceByName retrieves a specific service by its name.
//
// The equivalent WMI query is:
//
//	SELECT [selectors] FROM Win32_Service
//
// Refer to https://learn.microsoft.com/en-us/windows/win32/cimwin32prov/win32-service
// for the WMI class definition.
func QueryServiceByName(name string, selectorList []string) (*cimv2.Win32_Service, error) {
	serviceQuery := query.NewWmiQueryWithSelectList("Win32_Service", selectorList, "Name", name)
	instances, err := QueryInstances("", serviceQuery)
	if err != nil {
		return nil, err
	}

	service, err := cimv2.NewWin32_ServiceEx1(instances[0])
	if err != nil {
		return nil, fmt.Errorf("failed to get service %s: %w", name, err)
	}

	return service, err
}

// GetServiceName returns the name of a service.
func GetServiceName(service ServiceInterface) (string, error) {
	return service.GetPropertyName()
}

// GetServiceDisplayName returns the display name of a service.
func GetServiceDisplayName(service ServiceInterface) (string, error) {
	return service.GetPropertyDisplayName()
}

// GetServiceState returns the state of a service.
func GetServiceState(service ServiceInterface) (string, error) {
	return service.GetPropertyState()
}

// GetServiceStartMode returns the start mode of a service.
func GetServiceStartMode(service ServiceInterface) (string, error) {
	return service.GetPropertyStartMode()
}

// Win32Service wraps the WMI class Win32_Service (mainly for testing)
type Win32Service struct {
	*cimv2.Win32_Service
}

func (s *Win32Service) GetDependents() ([]ServiceInterface, error) {
	collection, err := s.GetAssociated("Win32_DependentService", "Win32_Service", "Dependent", "Antecedent")
	if err != nil {
		return nil, err
	}

	var result []ServiceInterface
	for _, coll := range collection {
		service, err := cimv2.NewWin32_ServiceEx1(coll)
		if err != nil {
			return nil, err
		}

		result = append(result, &Win32Service{
			service,
		})
	}
	return result, nil
}

type Win32ServiceFactory struct {
}

func (impl Win32ServiceFactory) GetService(name string) (ServiceInterface, error) {
	service, err := QueryServiceByName(name, ServiceSelectorList)
	if err != nil {
		return nil, err
	}

	return &Win32Service{Win32_Service: service}, nil
}

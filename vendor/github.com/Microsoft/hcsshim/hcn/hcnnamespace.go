package hcn

import (
	"encoding/json"
	"os"
	"syscall"

	"github.com/Microsoft/go-winio/pkg/guid"
	icni "github.com/Microsoft/hcsshim/internal/cni"
	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/Microsoft/hcsshim/internal/regstate"
	"github.com/Microsoft/hcsshim/internal/runhcs"
	"github.com/sirupsen/logrus"
)

// NamespaceResourceEndpoint represents an Endpoint attached to a Namespace.
type NamespaceResourceEndpoint struct {
	Id string `json:"ID,"`
}

// NamespaceResourceContainer represents a Container attached to a Namespace.
type NamespaceResourceContainer struct {
	Id string `json:"ID,"`
}

// NamespaceResourceType determines whether the Namespace resource is a Container or Endpoint.
type NamespaceResourceType string

var (
	// NamespaceResourceTypeContainer are contianers associated with a Namespace.
	NamespaceResourceTypeContainer NamespaceResourceType = "Container"
	// NamespaceResourceTypeEndpoint are endpoints associated with a Namespace.
	NamespaceResourceTypeEndpoint NamespaceResourceType = "Endpoint"
)

// NamespaceResource is associated with a namespace
type NamespaceResource struct {
	Type NamespaceResourceType `json:","` // Container, Endpoint
	Data json.RawMessage       `json:","`
}

// NamespaceType determines whether the Namespace is for a Host or Guest
type NamespaceType string

var (
	// NamespaceTypeHost are host namespaces.
	NamespaceTypeHost NamespaceType = "Host"
	// NamespaceTypeHostDefault are host namespaces in the default compartment.
	NamespaceTypeHostDefault NamespaceType = "HostDefault"
	// NamespaceTypeGuest are guest namespaces.
	NamespaceTypeGuest NamespaceType = "Guest"
	// NamespaceTypeGuestDefault are guest namespaces in the default compartment.
	NamespaceTypeGuestDefault NamespaceType = "GuestDefault"
)

// HostComputeNamespace represents a namespace (AKA compartment) in
type HostComputeNamespace struct {
	Id            string              `json:"ID,omitempty"`
	NamespaceId   uint32              `json:",omitempty"`
	Type          NamespaceType       `json:",omitempty"` // Host, HostDefault, Guest, GuestDefault
	Resources     []NamespaceResource `json:",omitempty"`
	SchemaVersion SchemaVersion       `json:",omitempty"`
}

// ModifyNamespaceSettingRequest is the structure used to send request to modify a namespace.
// Used to Add/Remove an endpoints and containers to/from a namespace.
type ModifyNamespaceSettingRequest struct {
	ResourceType NamespaceResourceType `json:",omitempty"` // Container, Endpoint
	RequestType  RequestType           `json:",omitempty"` // Add, Remove, Update, Refresh
	Settings     json.RawMessage       `json:",omitempty"`
}

func getNamespace(namespaceGuid guid.GUID, query string) (*HostComputeNamespace, error) {
	// Open namespace.
	var (
		namespaceHandle  hcnNamespace
		resultBuffer     *uint16
		propertiesBuffer *uint16
	)
	hr := hcnOpenNamespace(&namespaceGuid, &namespaceHandle, &resultBuffer)
	if err := checkForErrors("hcnOpenNamespace", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query namespace.
	hr = hcnQueryNamespaceProperties(namespaceHandle, query, &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryNamespaceProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close namespace.
	hr = hcnCloseNamespace(namespaceHandle)
	if err := checkForErrors("hcnCloseNamespace", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to HostComputeNamespace
	var outputNamespace HostComputeNamespace
	if err := json.Unmarshal([]byte(properties), &outputNamespace); err != nil {
		return nil, err
	}
	return &outputNamespace, nil
}

func enumerateNamespaces(query string) ([]HostComputeNamespace, error) {
	// Enumerate all Namespace Guids
	var (
		resultBuffer    *uint16
		namespaceBuffer *uint16
	)
	hr := hcnEnumerateNamespaces(query, &namespaceBuffer, &resultBuffer)
	if err := checkForErrors("hcnEnumerateNamespaces", hr, resultBuffer); err != nil {
		return nil, err
	}

	namespaces := interop.ConvertAndFreeCoTaskMemString(namespaceBuffer)
	var namespaceIds []guid.GUID
	if err := json.Unmarshal([]byte(namespaces), &namespaceIds); err != nil {
		return nil, err
	}

	var outputNamespaces []HostComputeNamespace
	for _, namespaceGuid := range namespaceIds {
		namespace, err := getNamespace(namespaceGuid, query)
		if err != nil {
			return nil, err
		}
		outputNamespaces = append(outputNamespaces, *namespace)
	}
	return outputNamespaces, nil
}

func createNamespace(settings string) (*HostComputeNamespace, error) {
	// Create new namespace.
	var (
		namespaceHandle  hcnNamespace
		resultBuffer     *uint16
		propertiesBuffer *uint16
	)
	namespaceGuid := guid.GUID{}
	hr := hcnCreateNamespace(&namespaceGuid, settings, &namespaceHandle, &resultBuffer)
	if err := checkForErrors("hcnCreateNamespace", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query namespace.
	hcnQuery := defaultQuery()
	query, err := json.Marshal(hcnQuery)
	if err != nil {
		return nil, err
	}
	hr = hcnQueryNamespaceProperties(namespaceHandle, string(query), &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryNamespaceProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close namespace.
	hr = hcnCloseNamespace(namespaceHandle)
	if err := checkForErrors("hcnCloseNamespace", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to HostComputeNamespace
	var outputNamespace HostComputeNamespace
	if err := json.Unmarshal([]byte(properties), &outputNamespace); err != nil {
		return nil, err
	}
	return &outputNamespace, nil
}

func modifyNamespace(namespaceId string, settings string) (*HostComputeNamespace, error) {
	namespaceGuid, err := guid.FromString(namespaceId)
	if err != nil {
		return nil, errInvalidNamespaceID
	}
	// Open namespace.
	var (
		namespaceHandle  hcnNamespace
		resultBuffer     *uint16
		propertiesBuffer *uint16
	)
	hr := hcnOpenNamespace(&namespaceGuid, &namespaceHandle, &resultBuffer)
	if err := checkForErrors("hcnOpenNamespace", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Modify namespace.
	hr = hcnModifyNamespace(namespaceHandle, settings, &resultBuffer)
	if err := checkForErrors("hcnModifyNamespace", hr, resultBuffer); err != nil {
		return nil, err
	}
	// Query namespace.
	hcnQuery := defaultQuery()
	query, err := json.Marshal(hcnQuery)
	if err != nil {
		return nil, err
	}
	hr = hcnQueryNamespaceProperties(namespaceHandle, string(query), &propertiesBuffer, &resultBuffer)
	if err := checkForErrors("hcnQueryNamespaceProperties", hr, resultBuffer); err != nil {
		return nil, err
	}
	properties := interop.ConvertAndFreeCoTaskMemString(propertiesBuffer)
	// Close namespace.
	hr = hcnCloseNamespace(namespaceHandle)
	if err := checkForErrors("hcnCloseNamespace", hr, nil); err != nil {
		return nil, err
	}
	// Convert output to Namespace
	var outputNamespace HostComputeNamespace
	if err := json.Unmarshal([]byte(properties), &outputNamespace); err != nil {
		return nil, err
	}
	return &outputNamespace, nil
}

func deleteNamespace(namespaceId string) error {
	namespaceGuid, err := guid.FromString(namespaceId)
	if err != nil {
		return errInvalidNamespaceID
	}
	var resultBuffer *uint16
	hr := hcnDeleteNamespace(&namespaceGuid, &resultBuffer)
	if err := checkForErrors("hcnDeleteNamespace", hr, resultBuffer); err != nil {
		return err
	}
	return nil
}

// ListNamespaces makes a call to list all available namespaces.
func ListNamespaces() ([]HostComputeNamespace, error) {
	hcnQuery := defaultQuery()
	namespaces, err := ListNamespacesQuery(hcnQuery)
	if err != nil {
		return nil, err
	}
	return namespaces, nil
}

// ListNamespacesQuery makes a call to query the list of available namespaces.
func ListNamespacesQuery(query HostComputeQuery) ([]HostComputeNamespace, error) {
	queryJson, err := json.Marshal(query)
	if err != nil {
		return nil, err
	}

	namespaces, err := enumerateNamespaces(string(queryJson))
	if err != nil {
		return nil, err
	}
	return namespaces, nil
}

// GetNamespaceByID returns the Namespace specified by Id.
func GetNamespaceByID(namespaceId string) (*HostComputeNamespace, error) {
	hcnQuery := defaultQuery()
	mapA := map[string]string{"ID": namespaceId}
	filter, err := json.Marshal(mapA)
	if err != nil {
		return nil, err
	}
	hcnQuery.Filter = string(filter)

	namespaces, err := ListNamespacesQuery(hcnQuery)
	if err != nil {
		return nil, err
	}
	if len(namespaces) == 0 {
		return nil, NamespaceNotFoundError{NamespaceID: namespaceId}
	}

	return &namespaces[0], err
}

// GetNamespaceEndpointIds returns the endpoints of the Namespace specified by Id.
func GetNamespaceEndpointIds(namespaceId string) ([]string, error) {
	namespace, err := GetNamespaceByID(namespaceId)
	if err != nil {
		return nil, err
	}
	var endpointsIds []string
	for _, resource := range namespace.Resources {
		if resource.Type == "Endpoint" {
			var endpointResource NamespaceResourceEndpoint
			if err := json.Unmarshal([]byte(resource.Data), &endpointResource); err != nil {
				return nil, err
			}
			endpointsIds = append(endpointsIds, endpointResource.Id)
		}
	}
	return endpointsIds, nil
}

// GetNamespaceContainerIds returns the containers of the Namespace specified by Id.
func GetNamespaceContainerIds(namespaceId string) ([]string, error) {
	namespace, err := GetNamespaceByID(namespaceId)
	if err != nil {
		return nil, err
	}
	var containerIds []string
	for _, resource := range namespace.Resources {
		if resource.Type == "Container" {
			var contaienrResource NamespaceResourceContainer
			if err := json.Unmarshal([]byte(resource.Data), &contaienrResource); err != nil {
				return nil, err
			}
			containerIds = append(containerIds, contaienrResource.Id)
		}
	}
	return containerIds, nil
}

// NewNamespace creates a new Namespace object
func NewNamespace(nsType NamespaceType) *HostComputeNamespace {
	return &HostComputeNamespace{
		Type:          nsType,
		SchemaVersion: V2SchemaVersion(),
	}
}

// Create Namespace.
func (namespace *HostComputeNamespace) Create() (*HostComputeNamespace, error) {
	logrus.Debugf("hcn::HostComputeNamespace::Create id=%s", namespace.Id)

	jsonString, err := json.Marshal(namespace)
	if err != nil {
		return nil, err
	}

	logrus.Debugf("hcn::HostComputeNamespace::Create JSON: %s", jsonString)
	namespace, hcnErr := createNamespace(string(jsonString))
	if hcnErr != nil {
		return nil, hcnErr
	}
	return namespace, nil
}

// Delete Namespace.
func (namespace *HostComputeNamespace) Delete() error {
	logrus.Debugf("hcn::HostComputeNamespace::Delete id=%s", namespace.Id)

	if err := deleteNamespace(namespace.Id); err != nil {
		return err
	}
	return nil
}

// Sync Namespace endpoints with the appropriate sandbox container holding the
// network namespace open. If no sandbox container is found for this namespace
// this method is determined to be a success and will not return an error in
// this case. If the sandbox container is found and a sync is initiated any
// failures will be returned via this method.
//
// This call initiates a sync between endpoints and the matching UtilityVM
// hosting those endpoints. It is safe to call for any `NamespaceType` but
// `NamespaceTypeGuest` is the only case when a sync will actually occur. For
// `NamespaceTypeHost` the process container will be automatically synchronized
// when the the endpoint is added via `AddNamespaceEndpoint`.
//
// Note: This method sync's both additions and removals of endpoints from a
// `NamespaceTypeGuest` namespace.
func (namespace *HostComputeNamespace) Sync() error {
	logrus.WithField("id", namespace.Id).Debugf("hcs::HostComputeNamespace::Sync")

	// We only attempt a sync for namespace guest.
	if namespace.Type != NamespaceTypeGuest {
		return nil
	}

	// Look in the registry for the key to map from namespace id to pod-id
	cfg, err := icni.LoadPersistedNamespaceConfig(namespace.Id)
	if err != nil {
		if regstate.IsNotFoundError(err) {
			return nil
		}
		return err
	}
	req := runhcs.VMRequest{
		ID: cfg.ContainerID,
		Op: runhcs.OpSyncNamespace,
	}
	shimPath := runhcs.VMPipePath(cfg.HostUniqueID)
	if err := runhcs.IssueVMRequest(shimPath, &req); err != nil {
		// The shim is likey gone. Simply ignore the sync as if it didn't exist.
		if perr, ok := err.(*os.PathError); ok && perr.Err == syscall.ERROR_FILE_NOT_FOUND {
			// Remove the reg key there is no point to try again
			cfg.Remove()
			return nil
		}
		f := map[string]interface{}{
			"id":           namespace.Id,
			"container-id": cfg.ContainerID,
		}
		logrus.WithFields(f).
			WithError(err).
			Debugf("hcs::HostComputeNamespace::Sync failed to connect to shim pipe: '%s'", shimPath)
		return err
	}
	return nil
}

// ModifyNamespaceSettings updates the Endpoints/Containers of a Namespace.
func ModifyNamespaceSettings(namespaceId string, request *ModifyNamespaceSettingRequest) error {
	logrus.Debugf("hcn::HostComputeNamespace::ModifyNamespaceSettings id=%s", namespaceId)

	namespaceSettings, err := json.Marshal(request)
	if err != nil {
		return err
	}

	_, err = modifyNamespace(namespaceId, string(namespaceSettings))
	if err != nil {
		return err
	}
	return nil
}

// AddNamespaceEndpoint adds an endpoint to a Namespace.
func AddNamespaceEndpoint(namespaceId string, endpointId string) error {
	logrus.Debugf("hcn::HostComputeEndpoint::AddNamespaceEndpoint id=%s", endpointId)

	mapA := map[string]string{"EndpointId": endpointId}
	settingsJson, err := json.Marshal(mapA)
	if err != nil {
		return err
	}
	requestMessage := &ModifyNamespaceSettingRequest{
		ResourceType: NamespaceResourceTypeEndpoint,
		RequestType:  RequestTypeAdd,
		Settings:     settingsJson,
	}

	return ModifyNamespaceSettings(namespaceId, requestMessage)
}

// RemoveNamespaceEndpoint removes an endpoint from a Namespace.
func RemoveNamespaceEndpoint(namespaceId string, endpointId string) error {
	logrus.Debugf("hcn::HostComputeNamespace::RemoveNamespaceEndpoint id=%s", endpointId)

	mapA := map[string]string{"EndpointId": endpointId}
	settingsJson, err := json.Marshal(mapA)
	if err != nil {
		return err
	}
	requestMessage := &ModifyNamespaceSettingRequest{
		ResourceType: NamespaceResourceTypeEndpoint,
		RequestType:  RequestTypeRemove,
		Settings:     settingsJson,
	}

	return ModifyNamespaceSettings(namespaceId, requestMessage)
}

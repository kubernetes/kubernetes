package nodes

import (
	"fmt"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToNodeListQuery() (string, error)
	ToNodeListDetailQuery() (string, error)
}

// Provision state reports the current provision state of the node, these are only used in filtering
type ProvisionState string

const (
	Enroll       ProvisionState = "enroll"
	Verifying    ProvisionState = "verifying"
	Manageable   ProvisionState = "manageable"
	Available    ProvisionState = "available"
	Active       ProvisionState = "active"
	DeployWait   ProvisionState = "wait call-back"
	Deploying    ProvisionState = "deploying"
	DeployFail   ProvisionState = "deploy failed"
	DeployDone   ProvisionState = "deploy complete"
	Deleting     ProvisionState = "deleting"
	Deleted      ProvisionState = "deleted"
	Cleaning     ProvisionState = "cleaning"
	CleanWait    ProvisionState = "clean wait"
	CleanFail    ProvisionState = "clean failed"
	Error        ProvisionState = "error"
	Rebuild      ProvisionState = "rebuild"
	Inspecting   ProvisionState = "inspecting"
	InspectFail  ProvisionState = "inspect failed"
	InspectWait  ProvisionState = "inspect wait"
	Adopting     ProvisionState = "adopting"
	AdoptFail    ProvisionState = "adopt failed"
	Rescue       ProvisionState = "rescue"
	RescueFail   ProvisionState = "rescue failed"
	Rescuing     ProvisionState = "rescuing"
	UnrescueFail ProvisionState = "unrescue failed"
)

// TargetProvisionState is used when setting the provision state for a node.
type TargetProvisionState string

const (
	TargetActive   TargetProvisionState = "active"
	TargetDeleted  TargetProvisionState = "deleted"
	TargetManage   TargetProvisionState = "manage"
	TargetProvide  TargetProvisionState = "provide"
	TargetInspect  TargetProvisionState = "inspect"
	TargetAbort    TargetProvisionState = "abort"
	TargetClean    TargetProvisionState = "clean"
	TargetAdopt    TargetProvisionState = "adopt"
	TargetRescue   TargetProvisionState = "rescue"
	TargetUnrescue TargetProvisionState = "unrescue"
)

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the node attributes you want to see returned. Marker and Limit are used
// for pagination.
type ListOpts struct {
	// Filter the list by specific instance UUID
	InstanceUUID string `q:"instance_uuid"`

	// Filter the list by chassis UUID
	ChassisUUID string `q:"chassis_uuid"`

	// Filter the list by maintenance set to True or False
	Maintenance bool `q:"maintenance"`

	// Nodes which are, or are not, associated with an instance_uuid.
	Associated bool `q:"associated"`

	// Only return those with the specified provision_state.
	ProvisionState ProvisionState `q:"provision_state"`

	// Filter the list with the specified driver.
	Driver string `q:"driver"`

	// Filter the list with the specified resource class.
	ResourceClass string `q:"resource_class"`

	// Filter the list with the specified conductor_group.
	ConductorGroup string `q:"conductor_group"`

	// Filter the list with the specified fault.
	Fault string `q:"fault"`

	// One or more fields to be returned in the response.
	Fields []string `q:"fields"`

	// Requests a page size of items.
	Limit int `q:"limit"`

	// The ID of the last-seen item.
	Marker string `q:"marker"`

	// Sorts the response by the requested sort direction.
	SortDir string `q:"sort_dir"`

	// Sorts the response by the this attribute value.
	SortKey string `q:"sort_key"`

	// A string or UUID of the tenant who owns the baremetal node.
	Owner string `q:"owner"`
}

// ToNodeListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToNodeListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List makes a request against the API to list nodes accessible to you.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToNodeListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return NodePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// ToNodeListDetailQuery formats a ListOpts into a query string for the list details API.
func (opts ListOpts) ToNodeListDetailQuery() (string, error) {
	// Detail endpoint can't filter by Fields
	if len(opts.Fields) > 0 {
		return "", fmt.Errorf("fields is not a valid option when getting a detailed listing of nodes")
	}

	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// Return a list of bare metal Nodes with complete details. Some filtering is possible by passing in flags in ListOpts,
// but you cannot limit by the fields returned.
func ListDetail(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	// This URL is deprecated. In the future, we should compare the microversion and if >= 1.43, hit the listURL
	// with ListOpts{Detail: true,}
	url := listDetailURL(client)
	if opts != nil {
		query, err := opts.ToNodeListDetailQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return NodePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get requests details on a single node, by ID.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToNodeCreateMap() (map[string]interface{}, error)
}

// CreateOpts specifies node creation parameters.
type CreateOpts struct {
	// The boot interface for a Node, e.g. “pxe”.
	BootInterface string `json:"boot_interface,omitempty"`

	// The conductor group for a node. Case-insensitive string up to 255 characters, containing a-z, 0-9, _, -, and ..
	ConductorGroup string `json:"conductor_group,omitempty"`

	// The console interface for a node, e.g. “no-console”.
	ConsoleInterface string `json:"console_interface,omitempty"`

	// The deploy interface for a node, e.g. “iscsi”.
	DeployInterface string `json:"deploy_interface,omitempty"`

	// All the metadata required by the driver to manage this Node. List of fields varies between drivers, and can
	// be retrieved from the /v1/drivers/<DRIVER_NAME>/properties resource.
	DriverInfo map[string]interface{} `json:"driver_info,omitempty"`

	// name of the driver used to manage this Node.
	Driver string `json:"driver,omitempty"`

	// A set of one or more arbitrary metadata key and value pairs.
	Extra map[string]interface{} `json:"extra,omitempty"`

	// The interface used for node inspection, e.g. “no-inspect”.
	InspectInterface string `json:"inspect_interface,omitempty"`

	// Interface for out-of-band node management, e.g. “ipmitool”.
	ManagementInterface string `json:"management_interface,omitempty"`

	// Human-readable identifier for the Node resource. May be undefined. Certain words are reserved.
	Name string `json:"name,omitempty"`

	// Which Network Interface provider to use when plumbing the network connections for this Node.
	NetworkInterface string `json:"network_interface,omitempty"`

	// Interface used for performing power actions on the node, e.g. “ipmitool”.
	PowerInterface string `json:"power_interface,omitempty"`

	// Physical characteristics of this Node. Populated during inspection, if performed. Can be edited via the REST
	// API at any time.
	Properties map[string]interface{} `json:"properties,omitempty"`

	// Interface used for configuring RAID on this node, e.g. “no-raid”.
	RAIDInterface string `json:"raid_interface,omitempty"`

	// The interface used for node rescue, e.g. “no-rescue”.
	RescueInterface string `json:"rescue_interface,omitempty"`

	// A string which can be used by external schedulers to identify this Node as a unit of a specific type
	// of resource.
	ResourceClass string `json:"resource_class,omitempty"`

	// Interface used for attaching and detaching volumes on this node, e.g. “cinder”.
	StorageInterface string `json:"storage_interface,omitempty"`

	// The UUID for the resource.
	UUID string `json:"uuid,omitempty"`

	// Interface for vendor-specific functionality on this node, e.g. “no-vendor”.
	VendorInterface string `json:"vendor_interface,omitempty"`

	// A string or UUID of the tenant who owns the baremetal node.
	Owner string `json:"owner,omitempty"`
}

// ToNodeCreateMap assembles a request body based on the contents of a CreateOpts.
func (opts CreateOpts) ToNodeCreateMap() (map[string]interface{}, error) {
	body, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	return body, nil
}

// Create requests a node to be created
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	reqBody, err := opts.ToNodeCreateMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Post(createURL(client), reqBody, &r.Body, nil)
	return
}

type Patch interface {
	ToNodeUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts is a slice of Patches used to update a node
type UpdateOpts []Patch

type UpdateOp string

const (
	ReplaceOp UpdateOp = "replace"
	AddOp     UpdateOp = "add"
	RemoveOp  UpdateOp = "remove"
)

type UpdateOperation struct {
	Op    UpdateOp    `json:"op" required:"true"`
	Path  string      `json:"path" required:"true"`
	Value interface{} `json:"value,omitempty"`
}

func (opts UpdateOperation) ToNodeUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "")
}

// Update requests that a node be updated
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOpts) (r UpdateResult) {
	body := make([]map[string]interface{}, len(opts))
	for i, patch := range opts {
		result, err := patch.ToNodeUpdateMap()
		if err != nil {
			r.Err = err
			return
		}

		body[i] = result
	}
	_, r.Err = client.Patch(updateURL(client, id), body, &r.Body, &gophercloud.RequestOpts{
		JSONBody: &body,
		OkCodes:  []int{200},
	})
	return
}

// Delete requests that a node be removed
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// Request that Ironic validate whether the Node’s driver has enough information to manage the Node. This polls each
// interface on the driver, and returns the status of that interface.
func Validate(client *gophercloud.ServiceClient, id string) (r ValidateResult) {
	_, r.Err = client.Get(validateURL(client, id), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Inject NMI (Non-Masking Interrupts) for the given Node. This feature can be used for hardware diagnostics, and
// actual support depends on a driver.
func InjectNMI(client *gophercloud.ServiceClient, id string) (r InjectNMIResult) {
	_, r.Err = client.Put(injectNMIURL(client, id), map[string]string{}, nil, &gophercloud.RequestOpts{
		OkCodes: []int{204},
	})
	return
}

type BootDeviceOpts struct {
	BootDevice string `json:"boot_device"` // e.g., 'pxe', 'disk', etc.
	Persistent bool   `json:"persistent"`  // Whether this is one-time or not
}

// BootDeviceOptsBuilder allows extensions to add additional parameters to the
// SetBootDevice request.
type BootDeviceOptsBuilder interface {
	ToBootDeviceMap() (map[string]interface{}, error)
}

// ToBootDeviceSetMap assembles a request body based on the contents of a BootDeviceOpts.
func (opts BootDeviceOpts) ToBootDeviceMap() (map[string]interface{}, error) {
	body, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	return body, nil
}

// Set the boot device for the given Node, and set it persistently or for one-time boot. The exact behaviour
// of this depends on the hardware driver.
func SetBootDevice(client *gophercloud.ServiceClient, id string, bootDevice BootDeviceOptsBuilder) (r SetBootDeviceResult) {
	reqBody, err := bootDevice.ToBootDeviceMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Put(bootDeviceURL(client, id), reqBody, nil, &gophercloud.RequestOpts{
		OkCodes: []int{204},
	})
	return
}

// Get the current boot device for the given Node.
func GetBootDevice(client *gophercloud.ServiceClient, id string) (r BootDeviceResult) {
	_, r.Err = client.Get(bootDeviceURL(client, id), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Retrieve the acceptable set of supported boot devices for a specific Node.
func GetSupportedBootDevices(client *gophercloud.ServiceClient, id string) (r SupportedBootDeviceResult) {
	_, r.Err = client.Get(supportedBootDeviceURL(client, id), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// A cleaning step has required keys ‘interface’ and ‘step’, and optional key ‘args’. If specified,
// the value for ‘args’ is a keyword variable argument dictionary that is passed to the cleaning step
// method.
type CleanStep struct {
	Interface string            `json:"interface" required:"true"`
	Step      string            `json:"step" required:"true"`
	Args      map[string]string `json:"args,omitempty"`
}

// ProvisionStateOptsBuilder allows extensions to add additional parameters to the
// ChangeProvisionState request.
type ProvisionStateOptsBuilder interface {
	ToProvisionStateMap() (map[string]interface{}, error)
}

// Starting with Ironic API version 1.56, a configdrive may be a JSON object with structured data.
// Prior to this version, it must be a base64-encoded, gzipped ISO9660 image.
type ConfigDrive struct {
	MetaData    map[string]interface{} `json:"meta_data,omitempty"`
	NetworkData map[string]interface{} `json:"network_data,omitempty"`
	UserData    interface{}            `json:"user_data,omitempty"`
}

// ProvisionStateOpts for a request to change a node's provision state. A config drive should be base64-encoded
// gzipped ISO9660 image.
type ProvisionStateOpts struct {
	Target         TargetProvisionState `json:"target" required:"true"`
	ConfigDrive    interface{}          `json:"configdrive,omitempty"`
	CleanSteps     []CleanStep          `json:"clean_steps,omitempty"`
	RescuePassword string               `json:"rescue_password,omitempty"`
}

// ToProvisionStateMap assembles a request body based on the contents of a CreateOpts.
func (opts ProvisionStateOpts) ToProvisionStateMap() (map[string]interface{}, error) {
	body, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	return body, nil
}

// Request a change to the Node’s provision state. Acceptable target states depend on the Node’s current provision
// state. More detailed documentation of the Ironic State Machine is available in the developer docs.
func ChangeProvisionState(client *gophercloud.ServiceClient, id string, opts ProvisionStateOptsBuilder) (r ChangeStateResult) {
	reqBody, err := opts.ToProvisionStateMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Put(provisionStateURL(client, id), reqBody, nil, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	return
}

type TargetPowerState string

// TargetPowerState is used when changing the power state of a node.
const (
	PowerOn       TargetPowerState = "power on"
	PowerOff      TargetPowerState = "power off"
	Rebooting     TargetPowerState = "rebooting"
	SoftPowerOff  TargetPowerState = "soft power off"
	SoftRebooting TargetPowerState = "soft rebooting"
)

// PowerStateOptsBuilder allows extensions to add additional parameters to the ChangePowerState request.
type PowerStateOptsBuilder interface {
	ToPowerStateMap() (map[string]interface{}, error)
}

// PowerStateOpts for a request to change a node's power state.
type PowerStateOpts struct {
	Target  TargetPowerState `json:"target" required:"true"`
	Timeout int              `json:"timeout,omitempty"`
}

// ToPowerStateMap assembles a request body based on the contents of a PowerStateOpts.
func (opts PowerStateOpts) ToPowerStateMap() (map[string]interface{}, error) {
	body, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	return body, nil
}

// Request to change a Node's power state.
func ChangePowerState(client *gophercloud.ServiceClient, id string, opts PowerStateOptsBuilder) (r ChangePowerStateResult) {
	reqBody, err := opts.ToPowerStateMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Put(powerStateURL(client, id), reqBody, nil, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	return
}

// This is the desired RAID configuration on the bare metal node.
type RAIDConfigOpts struct {
	LogicalDisks []LogicalDisk `json:"logical_disks"`
}

// RAIDConfigOptsBuilder allows extensions to modify a set RAID config request.
type RAIDConfigOptsBuilder interface {
	ToRAIDConfigMap() (map[string]interface{}, error)
}

// RAIDLevel type is used to specify the RAID level for a logical disk.
type RAIDLevel string

const (
	RAID0  RAIDLevel = "0"
	RAID1  RAIDLevel = "1"
	RAID2  RAIDLevel = "2"
	RAID5  RAIDLevel = "5"
	RAID6  RAIDLevel = "6"
	RAID10 RAIDLevel = "1+0"
	RAID50 RAIDLevel = "5+0"
	RAID60 RAIDLevel = "6+0"
)

// DiskType is used to specify the disk type for a logical disk, e.g. hdd or ssd.
type DiskType string

const (
	HDD DiskType = "hdd"
	SSD DiskType = "ssd"
)

// InterfaceType is used to specify the interface for a logical disk.
type InterfaceType string

const (
	SATA DiskType = "sata"
	SCSI DiskType = "scsi"
	SAS  DiskType = "sas"
)

type LogicalDisk struct {
	// Size (Integer) of the logical disk to be created in GiB.  If unspecified, "MAX" will be used.
	SizeGB *int `json:"size_gb"`

	// RAID level for the logical disk.
	RAIDLevel RAIDLevel `json:"raid_level" required:"true"`

	// Name of the volume. Should be unique within the Node. If not specified, volume name will be auto-generated.
	VolumeName string `json:"volume_name,omitempty"`

	// Set to true if this is the root volume. At most one logical disk can have this set to true.
	IsRootVolume *bool `json:"is_root_volume,omitempty"`

	// Set to true if this logical disk can share physical disks with other logical disks.
	SharePhysicalDisks *bool `json:"share_physical_disks,omitempty"`

	// If this is not specified, disk type will not be a criterion to find backing physical disks
	DiskType DiskType `json:"disk_type,omitempty"`

	// If this is not specified, interface type will not be a criterion to find backing physical disks.
	InterfaceType InterfaceType `json:"interface_type,omitempty"`

	// Integer, number of disks to use for the logical disk. Defaults to minimum number of disks required
	// for the particular RAID level.
	NumberOfPhysicalDisks int `json:"number_of_physical_disks,omitempty"`

	// The name of the controller as read by the RAID interface.
	Controller string `json:"controller,omitempty"`

	// A list of physical disks to use as read by the RAID interface.
	PhysicalDisks []string `json:"physical_disks,omitempty"`
}

func (opts RAIDConfigOpts) ToRAIDConfigMap() (map[string]interface{}, error) {
	body, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	for _, v := range body["logical_disks"].([]interface{}) {
		if logicalDisk, ok := v.(map[string]interface{}); ok {
			if logicalDisk["size_gb"] == nil {
				logicalDisk["size_gb"] = "MAX"
			}
		}
	}

	return body, nil
}

// Request to change a Node's RAID config.
func SetRAIDConfig(client *gophercloud.ServiceClient, id string, raidConfigOptsBuilder RAIDConfigOptsBuilder) (r ChangeStateResult) {
	reqBody, err := raidConfigOptsBuilder.ToRAIDConfigMap()
	if err != nil {
		r.Err = err
		return
	}

	_, r.Err = client.Put(raidConfigURL(client, id), reqBody, nil, &gophercloud.RequestOpts{
		OkCodes: []int{204},
	})
	return
}

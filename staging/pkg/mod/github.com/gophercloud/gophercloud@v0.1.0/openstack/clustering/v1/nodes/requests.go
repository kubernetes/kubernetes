package nodes

import (
	"net/http"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToNodeCreateMap() (map[string]interface{}, error)
}

// CreateOpts represents options used to create a Node.
type CreateOpts struct {
	Role      string                 `json:"role,omitempty"`
	ProfileID string                 `json:"profile_id" required:"true"`
	ClusterID string                 `json:"cluster_id,omitempty"`
	Name      string                 `json:"name" required:"true"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// ToNodeCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToNodeCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "node")
}

// Create requests the creation of a new node.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToNodeCreateMap()
	if err != nil {
		r.Err = err
		return
	}

	var result *http.Response
	result, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToNodeUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represents options used to update a Node.
type UpdateOpts struct {
	Name      string                 `json:"name,omitempty"`
	ProfileID string                 `json:"profile_id,omitempty"`
	Role      string                 `json:"role,omitempty"`
	Metadata  map[string]interface{} `json:"metadata,omitempty"`
}

// ToNodeUpdateMap constructs a request body from UpdateOpts.
func (opts UpdateOpts) ToNodeUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "node")
}

// Update requests the update of a node.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToNodeUpdateMap()
	if err != nil {
		r.Err = err
		return
	}

	var result *http.Response
	result, r.Err = client.Patch(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

// ListOptsBuilder allows extensions to add additional parmeters to the
// List request.
type ListOptsBuilder interface {
	ToNodeListQuery() (string, error)
}

// ListOpts represents options used to list nodes.
type ListOpts struct {
	Limit         int    `q:"limit"`
	Marker        string `q:"marker"`
	Sort          string `q:"sort"`
	GlobalProject *bool  `q:"global_project"`
	ClusterID     string `q:"cluster_id"`
	Name          string `q:"name"`
	Status        string `q:"status"`
}

// ToNodeListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToNodeListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List instructs OpenStack to provide a list of nodes.
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

// Delete deletes the specified node.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	var result *http.Response
	result, r.Err = client.Delete(deleteURL(client, id), nil)
	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

// Get makes a request against senlin to get a details of a node type
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	var result *http.Response
	result, r.Err = client.Get(getURL(client, id), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

// OperationName represents valid values for node operation
type OperationName string

const (
	// Nova Profile Op Names
	RebootOperation         OperationName = "reboot"
	RebuildOperation        OperationName = "rebuild"
	ChangePasswordOperation OperationName = "change_password"
	PauseOperation          OperationName = "pause"
	UnpauseOperation        OperationName = "unpause"
	SuspendOperation        OperationName = "suspend"
	ResumeOperation         OperationName = "resume"
	LockOperation           OperationName = "lock"
	UnlockOperation         OperationName = "unlock"
	StartOperation          OperationName = "start"
	StopOperation           OperationName = "stop"
	RescueOperation         OperationName = "rescue"
	UnrescueOperation       OperationName = "unrescue"
	EvacuateOperation       OperationName = "evacuate"

	// Heat Pofile Op Names
	AbandonOperation OperationName = "abandon"
)

// ToNodeOperationMap constructs a request body from OperationOpts.
func (opts OperationOpts) ToNodeOperationMap() (map[string]interface{}, error) {
	optsMap := map[string]interface{}{string(opts.Operation): opts.Params}
	return optsMap, nil
}

// OperationOptsBuilder allows extensions to add additional parameters to the
// Op request.
type OperationOptsBuilder interface {
	ToNodeOperationMap() (map[string]interface{}, error)
}
type OperationParams map[string]interface{}

// OperationOpts represents options used to perform an operation on a node
type OperationOpts struct {
	Operation OperationName   `json:"operation" required:"true"`
	Params    OperationParams `json:"params,omitempty"`
}

func Ops(client *gophercloud.ServiceClient, id string, opts OperationOptsBuilder) (r ActionResult) {
	b, err := opts.ToNodeOperationMap()
	if err != nil {
		r.Err = err
		return
	}

	var result *http.Response
	result, r.Err = client.Post(opsURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

func (opts RecoverOpts) ToNodeRecoverMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "recover")
}

// RecoverAction represents valid values for recovering a node.
type RecoverAction string

const (
	RebootRecovery  RecoverAction = "REBOOT"
	RebuildRecovery RecoverAction = "REBUILD"
	// RECREATE currently is NOT supported. See https://github.com/openstack/senlin/blob/b30b2b8496b2b8af243ccd5292f38aec7a95664f/senlin/profiles/base.py#L533
	RecreateRecovery RecoverAction = "RECREATE"
)

type RecoverOpts struct {
	Operation RecoverAction `json:"operation,omitempty"`
	Check     *bool         `json:"check,omitempty"`
}

func Recover(client *gophercloud.ServiceClient, id string, opts RecoverOpts) (r ActionResult) {
	b, err := opts.ToNodeRecoverMap()
	if err != nil {
		r.Err = err
		return
	}
	var result *http.Response
	result, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	r.Header = result.Header
	return
}

func Check(client *gophercloud.ServiceClient, id string) (r ActionResult) {
	b := map[string]interface{}{
		"check": map[string]interface{}{},
	}

	var result *http.Response
	result, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

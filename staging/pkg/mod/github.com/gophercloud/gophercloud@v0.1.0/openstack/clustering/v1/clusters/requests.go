package clusters

import (
	"fmt"
	"net/http"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// AdjustmentType represents valid values for resizing a cluster.
type AdjustmentType string

const (
	ExactCapacityAdjustment      AdjustmentType = "EXACT_CAPACITY"
	ChangeInCapacityAdjustment   AdjustmentType = "CHANGE_IN_CAPACITY"
	ChangeInPercentageAdjustment AdjustmentType = "CHANGE_IN_PERCENTAGE"
)

// RecoveryAction represents valid values for recovering a cluster.
type RecoveryAction string

const (
	RebootRecovery   RecoveryAction = "REBOOT"
	RebuildRecovery  RecoveryAction = "REBUILD"
	RecreateRecovery RecoveryAction = "RECREATE"
)

// CreateOptsBuilder allows extensions to add additional parameters
// to the Create request.
type CreateOptsBuilder interface {
	ToClusterCreateMap() (map[string]interface{}, error)
}

// CreateOpts represents options used to create a cluster.
type CreateOpts struct {
	Name            string                 `json:"name" required:"true"`
	DesiredCapacity int                    `json:"desired_capacity"`
	ProfileID       string                 `json:"profile_id" required:"true"`
	MinSize         *int                   `json:"min_size,omitempty"`
	Timeout         int                    `json:"timeout,omitempty"`
	MaxSize         int                    `json:"max_size,omitempty"`
	Metadata        map[string]interface{} `json:"metadata,omitempty"`
	Config          map[string]interface{} `json:"config,omitempty"`
}

// ToClusterCreateMap constructs a request body from CreateOpts.
func (opts CreateOpts) ToClusterCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "cluster")
}

// Create requests the creation of a new cluster.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToClusterCreateMap()
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

// Get retrieves details of a single cluster.
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

// ListOptsBuilder allows extensions to add additional parameters to
// the List request.
type ListOptsBuilder interface {
	ToClusterListQuery() (string, error)
}

// ListOpts represents options to list clusters.
type ListOpts struct {
	Limit         int    `q:"limit"`
	Marker        string `q:"marker"`
	Sort          string `q:"sort"`
	GlobalProject *bool  `q:"global_project"`
	Name          string `q:"name,omitempty"`
	Status        string `q:"status,omitempty"`
}

// ToClusterListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToClusterListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List instructs OpenStack to provide a list of clusters.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToClusterListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ClusterPage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToClusterUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts represents options to update a cluster.
type UpdateOpts struct {
	Config      string                 `json:"config,omitempty"`
	Name        string                 `json:"name,omitempty"`
	ProfileID   string                 `json:"profile_id,omitempty"`
	Timeout     *int                   `json:"timeout,omitempty"`
	Metadata    map[string]interface{} `json:"metadata,omitempty"`
	ProfileOnly *bool                  `json:"profile_only,omitempty"`
}

// ToClusterUpdateMap assembles a request body based on the contents of
// UpdateOpts.
func (opts UpdateOpts) ToClusterUpdateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "cluster")
	if err != nil {
		return nil, err
	}
	return b, nil
}

// Update will update an existing cluster.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToClusterUpdateMap()
	if err != nil {
		r.Err = err
		return r
	}

	var result *http.Response
	result, r.Err = client.Patch(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 202},
	})

	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

// Delete deletes the specified cluster ID.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	var result *http.Response
	result, r.Err = client.Delete(deleteURL(client, id), nil)
	if r.Err == nil {
		r.Header = result.Header
	}
	return
}

// ResizeOptsBuilder allows extensions to add additional parameters to the
// resize request.
type ResizeOptsBuilder interface {
	ToClusterResizeMap() (map[string]interface{}, error)
}

// ResizeOpts represents options for resizing a cluster.
type ResizeOpts struct {
	AdjustmentType AdjustmentType `json:"adjustment_type,omitempty"`
	Number         interface{}    `json:"number,omitempty"`
	MinSize        *int           `json:"min_size,omitempty"`
	MaxSize        *int           `json:"max_size,omitempty"`
	MinStep        *int           `json:"min_step,omitempty"`
	Strict         *bool          `json:"strict,omitempty"`
}

// ToClusterResizeMap constructs a request body from ResizeOpts.
func (opts ResizeOpts) ToClusterResizeMap() (map[string]interface{}, error) {
	if opts.AdjustmentType != "" && opts.Number == nil {
		return nil, fmt.Errorf("Number field MUST NOT be empty when AdjustmentType field used")
	}

	switch opts.Number.(type) {
	case nil, int, int32, int64:
		// Valid type. Always allow
	case float32, float64:
		if opts.AdjustmentType != ChangeInPercentageAdjustment {
			return nil, fmt.Errorf("Only ChangeInPercentageAdjustment allows float value for Number field")
		}
	default:
		return nil, fmt.Errorf("Number field must be either int, float, or omitted")
	}

	return gophercloud.BuildRequestBody(opts, "resize")
}

func Resize(client *gophercloud.ServiceClient, id string, opts ResizeOptsBuilder) (r ActionResult) {
	b, err := opts.ToClusterResizeMap()
	if err != nil {
		r.Err = err
		return
	}

	var result *http.Response
	result, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

// ScaleInOptsBuilder allows extensions to add additional parameters to the
// ScaleIn request.
type ScaleInOptsBuilder interface {
	ToClusterScaleInMap() (map[string]interface{}, error)
}

// ScaleInOpts represents options used to scale-in a cluster.
type ScaleInOpts struct {
	Count *int `json:"count,omitempty"`
}

// ToClusterScaleInMap constructs a request body from ScaleInOpts.
func (opts ScaleInOpts) ToClusterScaleInMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "scale_in")
}

// ScaleIn will reduce the capacity of a cluster.
func ScaleIn(client *gophercloud.ServiceClient, id string, opts ScaleInOptsBuilder) (r ActionResult) {
	b, err := opts.ToClusterScaleInMap()
	if err != nil {
		r.Err = err
		return
	}
	var result *http.Response
	result, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

// ScaleOutOptsBuilder allows extensions to add additional parameters to the
// ScaleOut request.
type ScaleOutOptsBuilder interface {
	ToClusterScaleOutMap() (map[string]interface{}, error)
}

// ScaleOutOpts represents options used to scale-out a cluster.
type ScaleOutOpts struct {
	Count int `json:"count,omitempty"`
}

// ToClusterScaleOutMap constructs a request body from ScaleOutOpts.
func (opts ScaleOutOpts) ToClusterScaleOutMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "scale_out")
}

// ScaleOut will increase the capacity of a cluster.
func ScaleOut(client *gophercloud.ServiceClient, id string, opts ScaleOutOptsBuilder) (r ActionResult) {
	b, err := opts.ToClusterScaleOutMap()
	if err != nil {
		r.Err = err
		return
	}
	var result *http.Response
	result, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

// AttachPolicyOptsBuilder allows extensions to add additional parameters to the
// AttachPolicy request.
type AttachPolicyOptsBuilder interface {
	ToClusterAttachPolicyMap() (map[string]interface{}, error)
}

// PolicyOpts params
type AttachPolicyOpts struct {
	PolicyID string `json:"policy_id" required:"true"`
	Enabled  *bool  `json:"enabled,omitempty"`
}

// ToClusterAttachPolicyMap constructs a request body from AttachPolicyOpts.
func (opts AttachPolicyOpts) ToClusterAttachPolicyMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "policy_attach")
}

// Attach Policy will attach a policy to a cluster.
func AttachPolicy(client *gophercloud.ServiceClient, id string, opts AttachPolicyOptsBuilder) (r ActionResult) {
	b, err := opts.ToClusterAttachPolicyMap()
	if err != nil {
		r.Err = err
		return
	}
	var result *http.Response
	result, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

// UpdatePolicyOptsBuilder allows extensions to add additional parameters to the
// UpdatePolicy request.
type UpdatePolicyOptsBuilder interface {
	ToClusterUpdatePolicyMap() (map[string]interface{}, error)
}

// UpdatePolicyOpts represents options used to update a cluster policy.
type UpdatePolicyOpts struct {
	PolicyID string `json:"policy_id" required:"true"`
	Enabled  *bool  `json:"enabled,omitempty" required:"true"`
}

// ToClusterUpdatePolicyMap constructs a request body from UpdatePolicyOpts.
func (opts UpdatePolicyOpts) ToClusterUpdatePolicyMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "policy_update")
}

// UpdatePolicy will update a cluster's policy.
func UpdatePolicy(client *gophercloud.ServiceClient, id string, opts UpdatePolicyOptsBuilder) (r ActionResult) {
	b, err := opts.ToClusterUpdatePolicyMap()
	if err != nil {
		r.Err = err
		return
	}
	var result *http.Response
	result, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

// DetachPolicyOptsBuilder allows extensions to add additional parameters to the
// DetachPolicy request.
type DetachPolicyOptsBuilder interface {
	ToClusterDetachPolicyMap() (map[string]interface{}, error)
}

// DetachPolicyOpts represents options used to detach a policy from a cluster.
type DetachPolicyOpts struct {
	PolicyID string `json:"policy_id" required:"true"`
}

// ToClusterDetachPolicyMap constructs a request body from DetachPolicyOpts.
func (opts DetachPolicyOpts) ToClusterDetachPolicyMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "policy_detach")
}

// DetachPolicy will detach a policy from a cluster.
func DetachPolicy(client *gophercloud.ServiceClient, id string, opts DetachPolicyOptsBuilder) (r ActionResult) {
	b, err := opts.ToClusterDetachPolicyMap()
	if err != nil {
		r.Err = err
		return
	}

	var result *http.Response
	result, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

// ListPolicyOptsBuilder allows extensions to add additional parameters to the
// ListPolicies request.
type ListPoliciesOptsBuilder interface {
	ToClusterPoliciesListQuery() (string, error)
}

// ListPoliciesOpts represents options to list a cluster's policies.
type ListPoliciesOpts struct {
	Enabled *bool  `q:"enabled"`
	Name    string `q:"policy_name"`
	Type    string `q:"policy_type"`
	Sort    string `q:"sort"`
}

// ToClusterPoliciesListQuery formats a ListOpts into a query string.
func (opts ListPoliciesOpts) ToClusterPoliciesListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// ListPolicies instructs OpenStack to provide a list of policies for a cluster.
func ListPolicies(client *gophercloud.ServiceClient, clusterID string, opts ListPoliciesOptsBuilder) pagination.Pager {
	url := listPoliciesURL(client, clusterID)
	if opts != nil {
		query, err := opts.ToClusterPoliciesListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ClusterPolicyPage{pagination.SinglePageBase(r)}
	})
}

// GetPolicy retrieves details of a cluster policy.
func GetPolicy(client *gophercloud.ServiceClient, clusterID string, policyID string) (r GetPolicyResult) {
	_, r.Err = client.Get(getPolicyURL(client, clusterID, policyID), &r.Body, nil)
	return
}

// RecoverOptsBuilder allows extensions to add additional parameters to the
// Recover request.
type RecoverOptsBuilder interface {
	ToClusterRecoverMap() (map[string]interface{}, error)
}

// RecoverOpts represents options used to recover a cluster.
type RecoverOpts struct {
	Operation     RecoveryAction `json:"operation,omitempty"`
	Check         *bool          `json:"check,omitempty"`
	CheckCapacity *bool          `json:"check_capacity,omitempty"`
}

// ToClusterRecovermap constructs a request body from RecoverOpts.
func (opts RecoverOpts) ToClusterRecoverMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "recover")
}

// Recover implements cluster recover request.
func Recover(client *gophercloud.ServiceClient, id string, opts RecoverOptsBuilder) (r ActionResult) {
	b, err := opts.ToClusterRecoverMap()
	if err != nil {
		r.Err = err
		return
	}
	var result *http.Response
	result, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

// Check will perform a health check on a cluster.
func Check(client *gophercloud.ServiceClient, id string) (r ActionResult) {
	b := map[string]interface{}{
		"check": map[string]interface{}{},
	}

	var result *http.Response
	result, r.Err = client.Post(actionURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200, 201, 202},
	})
	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

// ToClusterCompleteLifecycleMap constructs a request body from CompleteLifecycleOpts.
func (opts CompleteLifecycleOpts) ToClusterCompleteLifecycleMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "complete_lifecycle")
}

type CompleteLifecycleOpts struct {
	LifecycleActionTokenID string `json:"lifecycle_action_token" required:"true"`
}

func CompleteLifecycle(client *gophercloud.ServiceClient, id string, opts CompleteLifecycleOpts) (r ActionResult) {
	b, err := opts.ToClusterCompleteLifecycleMap()
	if err != nil {
		r.Err = err
		return
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

func (opts AddNodesOpts) ToClusterAddNodeMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "add_nodes")
}

type AddNodesOpts struct {
	Nodes []string `json:"nodes" required:"true"`
}

func AddNodes(client *gophercloud.ServiceClient, id string, opts AddNodesOpts) (r ActionResult) {
	b, err := opts.ToClusterAddNodeMap()
	if err != nil {
		r.Err = err
		return
	}
	var result *http.Response
	result, r.Err = client.Post(nodeURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	r.Header = result.Header
	return
}

func (opts RemoveNodesOpts) ToClusterRemoveNodeMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "del_nodes")
}

type RemoveNodesOpts struct {
	Nodes []string `json:"nodes" required:"true"`
}

func RemoveNodes(client *gophercloud.ServiceClient, clusterID string, opts RemoveNodesOpts) (r DeleteResult) {
	b, err := opts.ToClusterRemoveNodeMap()
	if err != nil {
		r.Err = err
		return
	}
	var result *http.Response
	result, r.Err = client.Post(nodeURL(client, clusterID), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	r.Header = result.Header
	return
}

func (opts ReplaceNodesOpts) ToClusterReplaceNodeMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "replace_nodes")
}

type ReplaceNodesOpts struct {
	Nodes map[string]string `json:"nodes" required:"true"`
}

func ReplaceNodes(client *gophercloud.ServiceClient, id string, opts ReplaceNodesOpts) (r ActionResult) {
	b, err := opts.ToClusterReplaceNodeMap()
	if err != nil {
		r.Err = err
		return
	}
	var result *http.Response
	result, r.Err = client.Post(nodeURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{202},
	})
	r.Header = result.Header
	return
}

type CollectOptsBuilder interface {
	ToClusterCollectMap() (string, error)
}

// CollectOpts represents options to collect attribute values across a cluster
type CollectOpts struct {
	Path string `q:"path" required:"true"`
}

func (opts CollectOpts) ToClusterCollectMap() (string, error) {
	return opts.Path, nil
}

// Collect instructs OpenStack to aggregate attribute values across a cluster
func Collect(client *gophercloud.ServiceClient, id string, opts CollectOptsBuilder) (r CollectResult) {
	query, err := opts.ToClusterCollectMap()
	if err != nil {
		r.Err = err
		return
	}

	var result *http.Response
	result, r.Err = client.Get(collectURL(client, id, query), &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})

	if r.Err == nil {
		r.Header = result.Header
	}

	return
}

// OperationName represents valid values for cluster operation
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

// ToClusterOperationMap constructs a request body from OperationOpts.
func (opts OperationOpts) ToClusterOperationMap() (map[string]interface{}, error) {
	operationArg := struct {
		Filters OperationFilters `json:"filters,omitempty"`
		Params  OperationParams  `json:"params,omitempty"`
	}{
		Filters: opts.Filters,
		Params:  opts.Params,
	}

	return gophercloud.BuildRequestBody(operationArg, string(opts.Operation))
}

// OperationOptsBuilder allows extensions to add additional parameters to the
// Op request.
type OperationOptsBuilder interface {
	ToClusterOperationMap() (map[string]interface{}, error)
}
type OperationFilters map[string]interface{}
type OperationParams map[string]interface{}

// OperationOpts represents options used to perform an operation on a cluster
type OperationOpts struct {
	Operation OperationName    `json:"operation" required:"true"`
	Filters   OperationFilters `json:"filters,omitempty"`
	Params    OperationParams  `json:"params,omitempty"`
}

func Ops(client *gophercloud.ServiceClient, id string, opts OperationOptsBuilder) (r ActionResult) {
	b, err := opts.ToClusterOperationMap()
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

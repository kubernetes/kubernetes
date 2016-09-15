package stacks

import (
	"strings"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToStackCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts struct {
	// The name of the stack. It must start with an alphabetic character.
	Name string `json:"stack_name" required:"true"`
	// A structure that contains either the template file or url. Call the
	// associated methods to extract the information relevant to send in a create request.
	TemplateOpts *Template `json:"-" required:"true"`
	// Enables or disables deletion of all stack resources when a stack
	// creation fails. Default is true, meaning all resources are not deleted when
	// stack creation fails.
	DisableRollback *bool `json:"disable_rollback,omitempty"`
	// A structure that contains details for the environment of the stack.
	EnvironmentOpts *Environment `json:"-"`
	// User-defined parameters to pass to the template.
	Parameters map[string]string `json:"parameters,omitempty"`
	// The timeout for stack creation in minutes.
	Timeout int `json:"timeout_mins,omitempty"`
	// A list of tags to assosciate with the Stack
	Tags []string `json:"-"`
}

// ToStackCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToStackCreateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	if err := opts.TemplateOpts.Parse(); err != nil {
		return nil, err
	}

	if err := opts.TemplateOpts.getFileContents(opts.TemplateOpts.Parsed, ignoreIfTemplate, true); err != nil {
		return nil, err
	}
	opts.TemplateOpts.fixFileRefs()
	b["template"] = string(opts.TemplateOpts.Bin)

	files := make(map[string]string)
	for k, v := range opts.TemplateOpts.Files {
		files[k] = v
	}

	if opts.EnvironmentOpts != nil {
		if err := opts.EnvironmentOpts.Parse(); err != nil {
			return nil, err
		}
		if err := opts.EnvironmentOpts.getRRFileContents(ignoreIfEnvironment); err != nil {
			return nil, err
		}
		opts.EnvironmentOpts.fixFileRefs()
		for k, v := range opts.EnvironmentOpts.Files {
			files[k] = v
		}
		b["environment"] = string(opts.EnvironmentOpts.Bin)
	}

	if len(files) > 0 {
		b["files"] = files
	}

	if opts.Tags != nil {
		b["tags"] = strings.Join(opts.Tags, ",")
	}

	return b, nil
}

// Create accepts a CreateOpts struct and creates a new stack using the values
// provided.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToStackCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(createURL(c), b, &r.Body, nil)
	return
}

// AdoptOptsBuilder is the interface options structs have to satisfy in order
// to be used in the Adopt function in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type AdoptOptsBuilder interface {
	ToStackAdoptMap() (map[string]interface{}, error)
}

// AdoptOpts is the common options struct used in this package's Adopt
// operation.
type AdoptOpts struct {
	// Existing resources data represented as a string to add to the
	// new stack. Data returned by Abandon could be provided as AdoptsStackData.
	AdoptStackData string `json:"adopt_stack_data" required:"true"`
	// The name of the stack. It must start with an alphabetic character.
	Name string `json:"stack_name" required:"true"`
	// A structure that contains either the template file or url. Call the
	// associated methods to extract the information relevant to send in a create request.
	TemplateOpts *Template `json:"-" required:"true"`
	// The timeout for stack creation in minutes.
	Timeout int `json:"timeout_mins,omitempty"`
	// A structure that contains either the template file or url. Call the
	// associated methods to extract the information relevant to send in a create request.
	//TemplateOpts *Template `json:"-" required:"true"`
	// Enables or disables deletion of all stack resources when a stack
	// creation fails. Default is true, meaning all resources are not deleted when
	// stack creation fails.
	DisableRollback *bool `json:"disable_rollback,omitempty"`
	// A structure that contains details for the environment of the stack.
	EnvironmentOpts *Environment `json:"-"`
	// User-defined parameters to pass to the template.
	Parameters map[string]string `json:"parameters,omitempty"`
}

// ToStackAdoptMap casts a CreateOpts struct to a map.
func (opts AdoptOpts) ToStackAdoptMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	if err := opts.TemplateOpts.Parse(); err != nil {
		return nil, err
	}

	if err := opts.TemplateOpts.getFileContents(opts.TemplateOpts.Parsed, ignoreIfTemplate, true); err != nil {
		return nil, err
	}
	opts.TemplateOpts.fixFileRefs()
	b["template"] = string(opts.TemplateOpts.Bin)

	files := make(map[string]string)
	for k, v := range opts.TemplateOpts.Files {
		files[k] = v
	}

	if opts.EnvironmentOpts != nil {
		if err := opts.EnvironmentOpts.Parse(); err != nil {
			return nil, err
		}
		if err := opts.EnvironmentOpts.getRRFileContents(ignoreIfEnvironment); err != nil {
			return nil, err
		}
		opts.EnvironmentOpts.fixFileRefs()
		for k, v := range opts.EnvironmentOpts.Files {
			files[k] = v
		}
		b["environment"] = string(opts.EnvironmentOpts.Bin)
	}

	if len(files) > 0 {
		b["files"] = files
	}

	return b, nil
}

// Adopt accepts an AdoptOpts struct and creates a new stack using the resources
// from another stack.
func Adopt(c *gophercloud.ServiceClient, opts AdoptOptsBuilder) (r AdoptResult) {
	b, err := opts.ToStackAdoptMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(adoptURL(c), b, &r.Body, nil)
	return
}

// SortDir is a type for specifying in which direction to sort a list of stacks.
type SortDir string

// SortKey is a type for specifying by which key to sort a list of stacks.
type SortKey string

var (
	// SortAsc is used to sort a list of stacks in ascending order.
	SortAsc SortDir = "asc"
	// SortDesc is used to sort a list of stacks in descending order.
	SortDesc SortDir = "desc"
	// SortName is used to sort a list of stacks by name.
	SortName SortKey = "name"
	// SortStatus is used to sort a list of stacks by status.
	SortStatus SortKey = "status"
	// SortCreatedAt is used to sort a list of stacks by date created.
	SortCreatedAt SortKey = "created_at"
	// SortUpdatedAt is used to sort a list of stacks by date updated.
	SortUpdatedAt SortKey = "updated_at"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToStackListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the network attributes you want to see returned. SortKey allows you to sort
// by a particular network attribute. SortDir sets the direction, and is either
// `asc' or `desc'. Marker and Limit are used for pagination.
type ListOpts struct {
	Status  string  `q:"status"`
	Name    string  `q:"name"`
	Marker  string  `q:"marker"`
	Limit   int     `q:"limit"`
	SortKey SortKey `q:"sort_keys"`
	SortDir SortDir `q:"sort_dir"`
}

// ToStackListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToStackListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List returns a Pager which allows you to iterate over a collection of
// stacks. It accepts a ListOpts struct, which allows you to filter and sort
// the returned collection for greater efficiency.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(c)
	if opts != nil {
		query, err := opts.ToStackListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	createPage := func(r pagination.PageResult) pagination.Page {
		return StackPage{pagination.SinglePageBase(r)}
	}
	return pagination.NewPager(c, url, createPage)
}

// Get retreives a stack based on the stack name and stack ID.
func Get(c *gophercloud.ServiceClient, stackName, stackID string) (r GetResult) {
	_, r.Err = c.Get(getURL(c, stackName, stackID), &r.Body, nil)
	return
}

// UpdateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the Update operation in this package.
type UpdateOptsBuilder interface {
	ToStackUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains the common options struct used in this package's Update
// operation.
type UpdateOpts struct {
	// A structure that contains either the template file or url. Call the
	// associated methods to extract the information relevant to send in a create request.
	TemplateOpts *Template `json:"-" required:"true"`
	// A structure that contains details for the environment of the stack.
	EnvironmentOpts *Environment `json:"-"`
	// User-defined parameters to pass to the template.
	Parameters map[string]string `json:"parameters,omitempty"`
	// The timeout for stack creation in minutes.
	Timeout int `json:"timeout_mins,omitempty"`
	// A list of tags to assosciate with the Stack
	Tags []string `json:"-"`
}

// ToStackUpdateMap casts a CreateOpts struct to a map.
func (opts UpdateOpts) ToStackUpdateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	if err := opts.TemplateOpts.Parse(); err != nil {
		return nil, err
	}

	if err := opts.TemplateOpts.getFileContents(opts.TemplateOpts.Parsed, ignoreIfTemplate, true); err != nil {
		return nil, err
	}
	opts.TemplateOpts.fixFileRefs()
	b["template"] = string(opts.TemplateOpts.Bin)

	files := make(map[string]string)
	for k, v := range opts.TemplateOpts.Files {
		files[k] = v
	}

	if opts.EnvironmentOpts != nil {
		if err := opts.EnvironmentOpts.Parse(); err != nil {
			return nil, err
		}
		if err := opts.EnvironmentOpts.getRRFileContents(ignoreIfEnvironment); err != nil {
			return nil, err
		}
		opts.EnvironmentOpts.fixFileRefs()
		for k, v := range opts.EnvironmentOpts.Files {
			files[k] = v
		}
		b["environment"] = string(opts.EnvironmentOpts.Bin)
	}

	if len(files) > 0 {
		b["files"] = files
	}

	if opts.Tags != nil {
		b["tags"] = strings.Join(opts.Tags, ",")
	}

	return b, nil
}

// Update accepts an UpdateOpts struct and updates an existing stack using the values
// provided.
func Update(c *gophercloud.ServiceClient, stackName, stackID string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToStackUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Put(updateURL(c, stackName, stackID), b, nil, nil)
	return
}

// Delete deletes a stack based on the stack name and stack ID.
func Delete(c *gophercloud.ServiceClient, stackName, stackID string) (r DeleteResult) {
	_, r.Err = c.Delete(deleteURL(c, stackName, stackID), nil)
	return
}

// PreviewOptsBuilder is the interface options structs have to satisfy in order
// to be used in the Preview operation in this package.
type PreviewOptsBuilder interface {
	ToStackPreviewMap() (map[string]interface{}, error)
}

// PreviewOpts contains the common options struct used in this package's Preview
// operation.
type PreviewOpts struct {
	// The name of the stack. It must start with an alphabetic character.
	Name string `json:"stack_name" required:"true"`
	// The timeout for stack creation in minutes.
	Timeout int `json:"timeout_mins" required:"true"`
	// A structure that contains either the template file or url. Call the
	// associated methods to extract the information relevant to send in a create request.
	TemplateOpts *Template `json:"-" required:"true"`
	// Enables or disables deletion of all stack resources when a stack
	// creation fails. Default is true, meaning all resources are not deleted when
	// stack creation fails.
	DisableRollback *bool `json:"disable_rollback,omitempty"`
	// A structure that contains details for the environment of the stack.
	EnvironmentOpts *Environment `json:"-"`
	// User-defined parameters to pass to the template.
	Parameters map[string]string `json:"parameters,omitempty"`
}

// ToStackPreviewMap casts a PreviewOpts struct to a map.
func (opts PreviewOpts) ToStackPreviewMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	if err := opts.TemplateOpts.Parse(); err != nil {
		return nil, err
	}

	if err := opts.TemplateOpts.getFileContents(opts.TemplateOpts.Parsed, ignoreIfTemplate, true); err != nil {
		return nil, err
	}
	opts.TemplateOpts.fixFileRefs()
	b["template"] = string(opts.TemplateOpts.Bin)

	files := make(map[string]string)
	for k, v := range opts.TemplateOpts.Files {
		files[k] = v
	}

	if opts.EnvironmentOpts != nil {
		if err := opts.EnvironmentOpts.Parse(); err != nil {
			return nil, err
		}
		if err := opts.EnvironmentOpts.getRRFileContents(ignoreIfEnvironment); err != nil {
			return nil, err
		}
		opts.EnvironmentOpts.fixFileRefs()
		for k, v := range opts.EnvironmentOpts.Files {
			files[k] = v
		}
		b["environment"] = string(opts.EnvironmentOpts.Bin)
	}

	if len(files) > 0 {
		b["files"] = files
	}

	return b, nil
}

// Preview accepts a PreviewOptsBuilder interface and creates a preview of a stack using the values
// provided.
func Preview(c *gophercloud.ServiceClient, opts PreviewOptsBuilder) (r PreviewResult) {
	b, err := opts.ToStackPreviewMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = c.Post(previewURL(c), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Abandon deletes the stack with the provided stackName and stackID, but leaves its
// resources intact, and returns data describing the stack and its resources.
func Abandon(c *gophercloud.ServiceClient, stackName, stackID string) (r AbandonResult) {
	_, r.Err = c.Delete(abandonURL(c, stackName, stackID), &gophercloud.RequestOpts{
		JSONResponse: &r.Body,
		OkCodes:      []int{200},
	})
	return
}

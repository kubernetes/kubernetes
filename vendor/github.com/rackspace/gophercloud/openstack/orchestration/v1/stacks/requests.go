package stacks

import (
	"errors"
	"strings"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// Rollback is used to specify whether or not a stack can be rolled back.
type Rollback *bool

var (
	disable = true
	// Disable is used to specify that a stack cannot be rolled back.
	Disable Rollback = &disable
	enable           = false
	// Enable is used to specify that a stack can be rolled back.
	Enable Rollback = &enable
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
	// (REQUIRED) The name of the stack. It must start with an alphabetic character.
	Name string
	// (REQUIRED) A structure that contains either the template file or url. Call the
	// associated methods to extract the information relevant to send in a create request.
	TemplateOpts *Template
	// (DEPRECATED): Please use TemplateOpts for providing the template. If
	// TemplateOpts is provided, TemplateURL will be ignored
	// (OPTIONAL; REQUIRED IF Template IS EMPTY) The URL of the template to instantiate.
	// This value is ignored if Template is supplied inline.
	TemplateURL string
	// (DEPRECATED): Please use TemplateOpts for providing the template. If
	// TemplateOpts is provided, Template will be ignored
	// (OPTIONAL; REQUIRED IF TemplateURL IS EMPTY) A template to instantiate. The value
	// is a stringified version of the JSON/YAML template. Since the template will likely
	// be located in a file, one way to set this variable is by using ioutil.ReadFile:
	// import "io/ioutil"
	// var opts stacks.CreateOpts
	// b, err := ioutil.ReadFile("path/to/you/template/file.json")
	// if err != nil {
	//   // handle error...
	// }
	// opts.Template = string(b)
	Template string
	// (OPTIONAL) Enables or disables deletion of all stack resources when a stack
	// creation fails. Default is true, meaning all resources are not deleted when
	// stack creation fails.
	DisableRollback Rollback
	// (OPTIONAL) A structure that contains details for the environment of the stack.
	EnvironmentOpts *Environment
	// (DEPRECATED): Please use EnvironmentOpts to provide Environment data
	// (OPTIONAL) A stringified JSON environment for the stack.
	Environment string
	// (DEPRECATED): Files is automatically determined
	// by parsing the template and environment passed as TemplateOpts and
	// EnvironmentOpts respectively.
	// (OPTIONAL) A map that maps file names to file contents. It can also be used
	// to pass provider template contents. Example:
	// Files: `{"myfile": "#!/bin/bash\necho 'Hello world' > /root/testfile.txt"}`
	Files map[string]interface{}
	// (OPTIONAL) User-defined parameters to pass to the template.
	Parameters map[string]string
	// (OPTIONAL) The timeout for stack creation in minutes.
	Timeout int
	// (OPTIONAL) A list of tags to assosciate with the Stack
	Tags []string
}

// ToStackCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToStackCreateMap() (map[string]interface{}, error) {
	s := make(map[string]interface{})

	if opts.Name == "" {
		return s, errors.New("Required field 'Name' not provided.")
	}
	s["stack_name"] = opts.Name
	Files := make(map[string]string)
	if opts.TemplateOpts == nil {
		if opts.Template != "" {
			s["template"] = opts.Template
		} else if opts.TemplateURL != "" {
			s["template_url"] = opts.TemplateURL
		} else {
			return s, errors.New("Either Template or TemplateURL must be provided.")
		}
	} else {
		if err := opts.TemplateOpts.Parse(); err != nil {
			return nil, err
		}

		if err := opts.TemplateOpts.getFileContents(opts.TemplateOpts.Parsed, ignoreIfTemplate, true); err != nil {
			return nil, err
		}
		opts.TemplateOpts.fixFileRefs()
		s["template"] = string(opts.TemplateOpts.Bin)

		for k, v := range opts.TemplateOpts.Files {
			Files[k] = v
		}
	}
	if opts.DisableRollback != nil {
		s["disable_rollback"] = &opts.DisableRollback
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
			Files[k] = v
		}
		s["environment"] = string(opts.EnvironmentOpts.Bin)
	} else if opts.Environment != "" {
		s["environment"] = opts.Environment
	}

	if opts.Files != nil {
		s["files"] = opts.Files
	} else {
		s["files"] = Files
	}

	if opts.DisableRollback != nil {
		s["disable_rollback"] = &opts.DisableRollback
	}

	if opts.Parameters != nil {
		s["parameters"] = opts.Parameters
	}

	if opts.Timeout != 0 {
		s["timeout_mins"] = opts.Timeout
	}

	if opts.Tags != nil {
		s["tags"] = strings.Join(opts.Tags, ",")
	}
	return s, nil
}

// Create accepts a CreateOpts struct and creates a new stack using the values
// provided.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToStackCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Post(createURL(c), reqBody, &res.Body, nil)
	return res
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
	// (REQUIRED) Existing resources data represented as a string to add to the
	// new stack. Data returned by Abandon could be provided as AdoptsStackData.
	AdoptStackData string
	// (REQUIRED) The name of the stack. It must start with an alphabetic character.
	Name string
	// (REQUIRED) The timeout for stack creation in minutes.
	Timeout int
	// (REQUIRED) A structure that contains either the template file or url. Call the
	// associated methods to extract the information relevant to send in a create request.
	TemplateOpts *Template
	// (DEPRECATED): Please use TemplateOpts for providing the template. If
	// TemplateOpts is provided, TemplateURL will be ignored
	// (OPTIONAL; REQUIRED IF Template IS EMPTY) The URL of the template to instantiate.
	// This value is ignored if Template is supplied inline.
	TemplateURL string
	// (DEPRECATED): Please use TemplateOpts for providing the template. If
	// TemplateOpts is provided, Template will be ignored
	// (OPTIONAL; REQUIRED IF TemplateURL IS EMPTY) A template to instantiate. The value
	// is a stringified version of the JSON/YAML template. Since the template will likely
	// be located in a file, one way to set this variable is by using ioutil.ReadFile:
	// import "io/ioutil"
	// var opts stacks.CreateOpts
	// b, err := ioutil.ReadFile("path/to/you/template/file.json")
	// if err != nil {
	//   // handle error...
	// }
	// opts.Template = string(b)
	Template string
	// (OPTIONAL) Enables or disables deletion of all stack resources when a stack
	// creation fails. Default is true, meaning all resources are not deleted when
	// stack creation fails.
	DisableRollback Rollback
	// (OPTIONAL) A structure that contains details for the environment of the stack.
	EnvironmentOpts *Environment
	// (DEPRECATED): Please use EnvironmentOpts to provide Environment data
	// (OPTIONAL) A stringified JSON environment for the stack.
	Environment string
	// (DEPRECATED): Files is automatically determined
	// by parsing the template and environment passed as TemplateOpts and
	// EnvironmentOpts respectively.
	// (OPTIONAL) A map that maps file names to file contents. It can also be used
	// to pass provider template contents. Example:
	// Files: `{"myfile": "#!/bin/bash\necho 'Hello world' > /root/testfile.txt"}`
	Files map[string]interface{}
	// (OPTIONAL) User-defined parameters to pass to the template.
	Parameters map[string]string
}

// ToStackAdoptMap casts a CreateOpts struct to a map.
func (opts AdoptOpts) ToStackAdoptMap() (map[string]interface{}, error) {
	s := make(map[string]interface{})

	if opts.Name == "" {
		return s, errors.New("Required field 'Name' not provided.")
	}
	s["stack_name"] = opts.Name
	Files := make(map[string]string)
	if opts.AdoptStackData != "" {
		s["adopt_stack_data"] = opts.AdoptStackData
	} else if opts.TemplateOpts == nil {
		if opts.Template != "" {
			s["template"] = opts.Template
		} else if opts.TemplateURL != "" {
			s["template_url"] = opts.TemplateURL
		} else {
			return s, errors.New("One of AdoptStackData, Template, TemplateURL or TemplateOpts must be provided.")
		}
	} else {
		if err := opts.TemplateOpts.Parse(); err != nil {
			return nil, err
		}

		if err := opts.TemplateOpts.getFileContents(opts.TemplateOpts.Parsed, ignoreIfTemplate, true); err != nil {
			return nil, err
		}
		opts.TemplateOpts.fixFileRefs()
		s["template"] = string(opts.TemplateOpts.Bin)

		for k, v := range opts.TemplateOpts.Files {
			Files[k] = v
		}
	}

	if opts.DisableRollback != nil {
		s["disable_rollback"] = &opts.DisableRollback
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
			Files[k] = v
		}
		s["environment"] = string(opts.EnvironmentOpts.Bin)
	} else if opts.Environment != "" {
		s["environment"] = opts.Environment
	}

	if opts.Files != nil {
		s["files"] = opts.Files
	} else {
		s["files"] = Files
	}

	if opts.Parameters != nil {
		s["parameters"] = opts.Parameters
	}

	if opts.Timeout != 0 {
		s["timeout"] = opts.Timeout
	}
	s["timeout_mins"] = opts.Timeout

	return s, nil
}

// Adopt accepts an AdoptOpts struct and creates a new stack using the resources
// from another stack.
func Adopt(c *gophercloud.ServiceClient, opts AdoptOptsBuilder) AdoptResult {
	var res AdoptResult

	reqBody, err := opts.ToStackAdoptMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Post(adoptURL(c), reqBody, &res.Body, nil)
	return res
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
func Get(c *gophercloud.ServiceClient, stackName, stackID string) GetResult {
	var res GetResult
	_, res.Err = c.Get(getURL(c, stackName, stackID), &res.Body, nil)
	return res
}

// UpdateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the Update operation in this package.
type UpdateOptsBuilder interface {
	ToStackUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contains the common options struct used in this package's Update
// operation.
type UpdateOpts struct {
	// (REQUIRED) A structure that contains either the template file or url. Call the
	// associated methods to extract the information relevant to send in a create request.
	TemplateOpts *Template
	// (DEPRECATED): Please use TemplateOpts for providing the template. If
	// TemplateOpts is provided, TemplateURL will be ignored
	// (OPTIONAL; REQUIRED IF Template IS EMPTY) The URL of the template to instantiate.
	// This value is ignored if Template is supplied inline.
	TemplateURL string
	// (DEPRECATED): Please use TemplateOpts for providing the template. If
	// TemplateOpts is provided, Template will be ignored
	// (OPTIONAL; REQUIRED IF TemplateURL IS EMPTY) A template to instantiate. The value
	// is a stringified version of the JSON/YAML template. Since the template will likely
	// be located in a file, one way to set this variable is by using ioutil.ReadFile:
	// import "io/ioutil"
	// var opts stacks.CreateOpts
	// b, err := ioutil.ReadFile("path/to/you/template/file.json")
	// if err != nil {
	//   // handle error...
	// }
	// opts.Template = string(b)
	Template string
	// (OPTIONAL) A structure that contains details for the environment of the stack.
	EnvironmentOpts *Environment
	// (DEPRECATED): Please use EnvironmentOpts to provide Environment data
	// (OPTIONAL) A stringified JSON environment for the stack.
	Environment string
	// (DEPRECATED): Files is automatically determined
	// by parsing the template and environment passed as TemplateOpts and
	// EnvironmentOpts respectively.
	// (OPTIONAL) A map that maps file names to file contents. It can also be used
	// to pass provider template contents. Example:
	// Files: `{"myfile": "#!/bin/bash\necho 'Hello world' > /root/testfile.txt"}`
	Files map[string]interface{}
	// (OPTIONAL) User-defined parameters to pass to the template.
	Parameters map[string]string
	// (OPTIONAL) The timeout for stack creation in minutes.
	Timeout int
	// (OPTIONAL) A list of tags to assosciate with the Stack
	Tags []string
}

// ToStackUpdateMap casts a CreateOpts struct to a map.
func (opts UpdateOpts) ToStackUpdateMap() (map[string]interface{}, error) {
	s := make(map[string]interface{})
	Files := make(map[string]string)
	if opts.TemplateOpts == nil {
		if opts.Template != "" {
			s["template"] = opts.Template
		} else if opts.TemplateURL != "" {
			s["template_url"] = opts.TemplateURL
		} else {
			return s, errors.New("Either Template or TemplateURL must be provided.")
		}
	} else {
		if err := opts.TemplateOpts.Parse(); err != nil {
			return nil, err
		}

		if err := opts.TemplateOpts.getFileContents(opts.TemplateOpts.Parsed, ignoreIfTemplate, true); err != nil {
			return nil, err
		}
		opts.TemplateOpts.fixFileRefs()
		s["template"] = string(opts.TemplateOpts.Bin)

		for k, v := range opts.TemplateOpts.Files {
			Files[k] = v
		}
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
			Files[k] = v
		}
		s["environment"] = string(opts.EnvironmentOpts.Bin)
	} else if opts.Environment != "" {
		s["environment"] = opts.Environment
	}

	if opts.Files != nil {
		s["files"] = opts.Files
	} else {
		s["files"] = Files
	}

	if opts.Parameters != nil {
		s["parameters"] = opts.Parameters
	}

	if opts.Timeout != 0 {
		s["timeout_mins"] = opts.Timeout
	}

	if opts.Tags != nil {
		s["tags"] = strings.Join(opts.Tags, ",")
	}

	return s, nil
}

// Update accepts an UpdateOpts struct and updates an existing stack using the values
// provided.
func Update(c *gophercloud.ServiceClient, stackName, stackID string, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult

	reqBody, err := opts.ToStackUpdateMap()
	if err != nil {
		res.Err = err
		return res
	}

	_, res.Err = c.Put(updateURL(c, stackName, stackID), reqBody, nil, nil)
	return res
}

// Delete deletes a stack based on the stack name and stack ID.
func Delete(c *gophercloud.ServiceClient, stackName, stackID string) DeleteResult {
	var res DeleteResult
	_, res.Err = c.Delete(deleteURL(c, stackName, stackID), nil)
	return res
}

// PreviewOptsBuilder is the interface options structs have to satisfy in order
// to be used in the Preview operation in this package.
type PreviewOptsBuilder interface {
	ToStackPreviewMap() (map[string]interface{}, error)
}

// PreviewOpts contains the common options struct used in this package's Preview
// operation.
type PreviewOpts struct {
	// (REQUIRED) The name of the stack. It must start with an alphabetic character.
	Name string
	// (REQUIRED) The timeout for stack creation in minutes.
	Timeout int
	// (REQUIRED) A structure that contains either the template file or url. Call the
	// associated methods to extract the information relevant to send in a create request.
	TemplateOpts *Template
	// (DEPRECATED): Please use TemplateOpts for providing the template. If
	// TemplateOpts is provided, TemplateURL will be ignored
	// (OPTIONAL; REQUIRED IF Template IS EMPTY) The URL of the template to instantiate.
	// This value is ignored if Template is supplied inline.
	TemplateURL string
	// (DEPRECATED): Please use TemplateOpts for providing the template. If
	// TemplateOpts is provided, Template will be ignored
	// (OPTIONAL; REQUIRED IF TemplateURL IS EMPTY) A template to instantiate. The value
	// is a stringified version of the JSON/YAML template. Since the template will likely
	// be located in a file, one way to set this variable is by using ioutil.ReadFile:
	// import "io/ioutil"
	// var opts stacks.CreateOpts
	// b, err := ioutil.ReadFile("path/to/you/template/file.json")
	// if err != nil {
	//   // handle error...
	// }
	// opts.Template = string(b)
	Template string
	// (OPTIONAL) Enables or disables deletion of all stack resources when a stack
	// creation fails. Default is true, meaning all resources are not deleted when
	// stack creation fails.
	DisableRollback Rollback
	// (OPTIONAL) A structure that contains details for the environment of the stack.
	EnvironmentOpts *Environment
	// (DEPRECATED): Please use EnvironmentOpts to provide Environment data
	// (OPTIONAL) A stringified JSON environment for the stack.
	Environment string
	// (DEPRECATED): Files is automatically determined
	// by parsing the template and environment passed as TemplateOpts and
	// EnvironmentOpts respectively.
	// (OPTIONAL) A map that maps file names to file contents. It can also be used
	// to pass provider template contents. Example:
	// Files: `{"myfile": "#!/bin/bash\necho 'Hello world' > /root/testfile.txt"}`
	Files map[string]interface{}
	// (OPTIONAL) User-defined parameters to pass to the template.
	Parameters map[string]string
}

// ToStackPreviewMap casts a PreviewOpts struct to a map.
func (opts PreviewOpts) ToStackPreviewMap() (map[string]interface{}, error) {
	s := make(map[string]interface{})

	if opts.Name == "" {
		return s, errors.New("Required field 'Name' not provided.")
	}
	s["stack_name"] = opts.Name
	Files := make(map[string]string)
	if opts.TemplateOpts == nil {
		if opts.Template != "" {
			s["template"] = opts.Template
		} else if opts.TemplateURL != "" {
			s["template_url"] = opts.TemplateURL
		} else {
			return s, errors.New("Either Template or TemplateURL must be provided.")
		}
	} else {
		if err := opts.TemplateOpts.Parse(); err != nil {
			return nil, err
		}

		if err := opts.TemplateOpts.getFileContents(opts.TemplateOpts.Parsed, ignoreIfTemplate, true); err != nil {
			return nil, err
		}
		opts.TemplateOpts.fixFileRefs()
		s["template"] = string(opts.TemplateOpts.Bin)

		for k, v := range opts.TemplateOpts.Files {
			Files[k] = v
		}
	}
	if opts.DisableRollback != nil {
		s["disable_rollback"] = &opts.DisableRollback
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
			Files[k] = v
		}
		s["environment"] = string(opts.EnvironmentOpts.Bin)
	} else if opts.Environment != "" {
		s["environment"] = opts.Environment
	}

	if opts.Files != nil {
		s["files"] = opts.Files
	} else {
		s["files"] = Files
	}

	if opts.Parameters != nil {
		s["parameters"] = opts.Parameters
	}

	if opts.Timeout != 0 {
		s["timeout_mins"] = opts.Timeout
	}

	return s, nil
}

// Preview accepts a PreviewOptsBuilder interface and creates a preview of a stack using the values
// provided.
func Preview(c *gophercloud.ServiceClient, opts PreviewOptsBuilder) PreviewResult {
	var res PreviewResult

	reqBody, err := opts.ToStackPreviewMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
	_, res.Err = c.Post(previewURL(c), reqBody, &res.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return res
}

// Abandon deletes the stack with the provided stackName and stackID, but leaves its
// resources intact, and returns data describing the stack and its resources.
func Abandon(c *gophercloud.ServiceClient, stackName, stackID string) AbandonResult {
	var res AbandonResult
	_, res.Err = c.Delete(abandonURL(c, stackName, stackID), &gophercloud.RequestOpts{
		JSONResponse: &res.Body,
		OkCodes:      []int{200},
	})
	return res
}

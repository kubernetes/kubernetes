package client

type Collection struct {
	Type         string                 `json:"type,omitempty"`
	ResourceType string                 `json:"resourceType,omitempty"`
	Links        map[string]string      `json:"links,omitempty"`
	CreateTypes  map[string]string      `json:"createTypes,omitempty"`
	Actions      map[string]string      `json:"actions,omitempty"`
	SortLinks    map[string]string      `json:"sortLinks,omitempty"`
	Pagination   *Pagination            `json:"pagination,omitempty"`
	Sort         *Sort                  `json:"sort,omitempty"`
	Filters      map[string][]Condition `json:"filters,omitempty"`
}

type GenericCollection struct {
	Collection
	Data []interface{} `json:"data,omitempty"`
}

type ResourceCollection struct {
	Collection
	Data []Resource `json:"data,omitempty"`
}

type Sort struct {
	Name    string `json:"name,omitempty"`
	Order   string `json:"order,omitempty"`
	Reverse string `json:"reverse,omitempty"`
}

type Condition struct {
	Modifier string      `json:"modifier,omitempty"`
	Value    interface{} `json:"value,omitempty"`
}

type Pagination struct {
	Marker   string `json:"marker,omitempty"`
	First    string `json:"first,omitempty"`
	Previous string `json:"previous,omitempty"`
	Next     string `json:"next,omitempty"`
	Limit    *int64 `json:"limit,omitempty"`
	Total    *int64 `json:"total,omitempty"`
	Partial  bool   `json:"partial,omitempty"`
}

type Resource struct {
	Id      string            `json:"id,omitempty"`
	Type    string            `json:"type,omitempty"`
	Links   map[string]string `json:"links"`
	Actions map[string]string `json:"actions"`
}

type Schema struct {
	Resource
	PluralName        string            `json:"pluralName,omitempty"`
	ResourceMethods   []string          `json:"resourceMethods,omitempty"`
	ResourceFields    map[string]Field  `json:"resourceFields,omitempty"`
	ResourceActions   map[string]Action `json:"resourceActions,omitempty"`
	CollectionMethods []string          `json:"collectionMethods,omitempty"`
	CollectionFields  map[string]Field  `json:"collectionFields,omitempty"`
	CollectionActions map[string]Action `json:"collectionActions,omitempty"`
	CollectionFilters map[string]Filter `json:"collectionFilters,omitempty"`
	IncludeableLinks  []string          `json:"includeableLinks,omitempty"`
}

type Field struct {
	Type         string      `json:"type,omitempty"`
	Default      interface{} `json:"default,omitempty"`
	Unique       bool        `json:"unique,omitempty"`
	Nullable     bool        `json:"nullable,omitempty"`
	Create       bool        `json:"create,omitempty"`
	Required     bool        `json:"required,omitempty"`
	Update       bool        `json:"update,omitempty"`
	MinLength    *int64      `json:"minLength,omitempty"`
	MaxLength    *int64      `json:"maxLength,omitempty"`
	Min          *int64      `json:"min,omitempty"`
	Max          *int64      `json:"max,omitempty"`
	Options      []string    `json:"options,omitempty"`
	ValidChars   string      `json:"validChars,omitempty"`
	InvalidChars string      `json:"invalidChars,omitempty"`
	Description  string      `json:"description,omitempty"`
}

type Action struct {
	Input  string `json:"input,omitempty"`
	Output string `json:"output,omitempty"`
}

type Filter struct {
	Modifiers []string `json:"modifiers,omitempty"`
}

type ListOpts struct {
	Filters map[string]interface{}
}

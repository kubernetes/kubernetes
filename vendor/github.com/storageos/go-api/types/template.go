package types

// Template is used to define an auto-naming rule.
type Template struct {

	// Template unique ID.
	// Read Only: true
	ID string `json:"id"`

	// Template name.
	// Required: true
	Name string `json:"name"`

	// Template description.
	Description string `json:"description"`

	// Template format.  This is used for pattern matching against labels.
	Format string `json:"format"`

	// Autoincrement defines whether there is a dynamic numeric component in the
	// template that must auto-increment when objects with the same name already
	// exists.
	AutoIncrement bool `json:"autoIncrement"`

	// Padding determines whether a dynamic numeric component in the name should
	// be padded.
	// default: false
	Padding bool `json:"padding"`

	// PaddingLength sets the length of the padding.  A Padding length of 3 would
	// set name similar to `abc001` for the first item.  Ignored if Padding set to
	// `false`.
	PaddingLength int `json:"paddingLength"`

	// Flag describing whether the template is active.
	// Default: false
	Active bool `json:"active"`

	// Weight is used to determine order during template processing.  Templates
	// with heavier weights are processed later.
	// default: 0
	Weight int `json:"weight"`

	// ObjectTypes defines the type names that the template can be applied to.
	ObjectTypes []string `json:"objectTypes"`

	// Labels define a list of the labels that the object must have in order for
	// the template to be applied.
	Labels map[string]string `json:"labels"`
}

// Templates is a collection of Template objects
type Templates []*Template

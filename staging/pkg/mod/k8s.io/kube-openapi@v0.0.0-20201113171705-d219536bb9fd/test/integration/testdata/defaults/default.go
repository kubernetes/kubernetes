package defaults

// +k8s:openapi-gen=true
type Defaulted struct {
	// +default="bar"
	Field string `json:"Field,omitempty"`
	// +default=0
	OtherField int
	// +default=["foo", "bar"]
	List []Item
	// +default={"s": "foo", "i": 5}
	Sub *SubStruct

	OtherSub SubStruct

	// +default={"foo": "bar"}
	Map map[string]Item
}

// +k8s:openapi-gen=true
type Item string

// +k8s:openapi-gen=true
type SubStruct struct {
	S string
	// +default=1
	I int `json:"I,omitempty"`
}

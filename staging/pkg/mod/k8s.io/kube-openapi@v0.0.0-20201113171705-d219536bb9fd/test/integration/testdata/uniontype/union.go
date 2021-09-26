package uniontype

// +k8s:openapi-gen=true
type TopLevelUnion struct {
	Name string `json:"name"`

	Union `json:",inline"`
}

// +k8s:openapi-gen=true
// +union
type Union struct {
	// +unionDiscriminator
	// +optional
	UnionType string `json:"unionType"`

	FieldA int `json:"fieldA,omitempty"`
	FieldB int `json:"fieldB,omitempty"`
}

// +k8s:openapi-gen=true
type Union2 struct {
	// +unionDiscriminator
	Type string `json:"type"`
	// +unionDeprecated
	Alpha int `json:"alpha,omitempty"`
	// +unionDeprecated
	Beta int `json:"beta,omitempty"`
}

// +k8s:openapi-gen=true
type InlinedUnion struct {
	Name string `json:"name"`

	// +unionDeprecated
	// +optional
	Field1 *int `json:"field1,omitempty"`
	// +unionDeprecated
	// +optional
	Field2 *int `json:"field2,omitempty"`

	Union  `json:",inline"`
	Union2 `json:",inline"`
}

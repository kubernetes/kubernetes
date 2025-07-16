package v1

import (
	"github.com/openshift/api/pkg/serialization"
	runtime "k8s.io/apimachinery/pkg/runtime"
)

var _ runtime.NestedObjectDecoder = &PolicyRule{}
var _ runtime.NestedObjectEncoder = &PolicyRule{}

func (c *PolicyRule) DecodeNestedObjects(d runtime.Decoder) error {
	// decoding failures result in a runtime.Unknown object being created in Object and passed
	// to conversion
	serialization.DecodeNestedRawExtensionOrUnknown(d, &c.AttributeRestrictions)
	return nil
}
func (c *PolicyRule) EncodeNestedObjects(e runtime.Encoder) error {
	return serialization.EncodeNestedRawExtension(e, &c.AttributeRestrictions)
}

var _ runtime.NestedObjectDecoder = &SelfSubjectRulesReview{}
var _ runtime.NestedObjectEncoder = &SelfSubjectRulesReview{}

func (c *SelfSubjectRulesReview) DecodeNestedObjects(d runtime.Decoder) error {
	// decoding failures result in a runtime.Unknown object being created in Object and passed
	// to conversion
	for i := range c.Status.Rules {
		c.Status.Rules[i].DecodeNestedObjects(d)
	}
	return nil
}
func (c *SelfSubjectRulesReview) EncodeNestedObjects(e runtime.Encoder) error {
	for i := range c.Status.Rules {
		if err := c.Status.Rules[i].EncodeNestedObjects(e); err != nil {
			return err
		}
	}
	return nil
}

var _ runtime.NestedObjectDecoder = &SubjectRulesReview{}
var _ runtime.NestedObjectEncoder = &SubjectRulesReview{}

func (c *SubjectRulesReview) DecodeNestedObjects(d runtime.Decoder) error {
	// decoding failures result in a runtime.Unknown object being created in Object and passed
	// to conversion
	for i := range c.Status.Rules {
		c.Status.Rules[i].DecodeNestedObjects(d)
	}
	return nil
}
func (c *SubjectRulesReview) EncodeNestedObjects(e runtime.Encoder) error {
	for i := range c.Status.Rules {
		if err := c.Status.Rules[i].EncodeNestedObjects(e); err != nil {
			return err
		}
	}
	return nil
}

var _ runtime.NestedObjectDecoder = &ClusterRole{}
var _ runtime.NestedObjectEncoder = &ClusterRole{}

func (c *ClusterRole) DecodeNestedObjects(d runtime.Decoder) error {
	// decoding failures result in a runtime.Unknown object being created in Object and passed
	// to conversion
	for i := range c.Rules {
		c.Rules[i].DecodeNestedObjects(d)
	}
	return nil
}
func (c *ClusterRole) EncodeNestedObjects(e runtime.Encoder) error {
	for i := range c.Rules {
		if err := c.Rules[i].EncodeNestedObjects(e); err != nil {
			return err
		}
	}
	return nil
}

var _ runtime.NestedObjectDecoder = &Role{}
var _ runtime.NestedObjectEncoder = &Role{}

func (c *Role) DecodeNestedObjects(d runtime.Decoder) error {
	// decoding failures result in a runtime.Unknown object being created in Object and passed
	// to conversion
	for i := range c.Rules {
		c.Rules[i].DecodeNestedObjects(d)
	}
	return nil
}
func (c *Role) EncodeNestedObjects(e runtime.Encoder) error {
	for i := range c.Rules {
		if err := c.Rules[i].EncodeNestedObjects(e); err != nil {
			return err
		}
	}
	return nil
}

var _ runtime.NestedObjectDecoder = &ClusterRoleList{}
var _ runtime.NestedObjectEncoder = &ClusterRoleList{}

func (c *ClusterRoleList) DecodeNestedObjects(d runtime.Decoder) error {
	// decoding failures result in a runtime.Unknown object being created in Object and passed
	// to conversion
	for i := range c.Items {
		c.Items[i].DecodeNestedObjects(d)
	}
	return nil
}
func (c *ClusterRoleList) EncodeNestedObjects(e runtime.Encoder) error {
	for i := range c.Items {
		if err := c.Items[i].EncodeNestedObjects(e); err != nil {
			return err
		}
	}
	return nil
}

var _ runtime.NestedObjectDecoder = &RoleList{}
var _ runtime.NestedObjectEncoder = &RoleList{}

func (c *RoleList) DecodeNestedObjects(d runtime.Decoder) error {
	// decoding failures result in a runtime.Unknown object being created in Object and passed
	// to conversion
	for i := range c.Items {
		c.Items[i].DecodeNestedObjects(d)
	}
	return nil
}
func (c *RoleList) EncodeNestedObjects(e runtime.Encoder) error {
	for i := range c.Items {
		if err := c.Items[i].EncodeNestedObjects(e); err != nil {
			return err
		}
	}
	return nil
}

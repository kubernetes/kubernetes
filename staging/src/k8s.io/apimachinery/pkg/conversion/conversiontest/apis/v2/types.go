package v2

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// To convert from v1 to v2, you have to either default the Type or write logic in your conversion method that effectively
// does the defaulting to decide what to do.  If you're migrating an older codebase, you need to make sure that nothing
// in the conversion path relies on having required fields set.  We used to simply allow a failure if that were the case,
// but if you need to tolerate it for APPLY you have code to inspect.
// In addition, you need to preserve the pointer aspect of Type through the internal hub to inform the internal->v2 conversion, which didn't use to be the case since defaulting used to happen when going v1->internal.
// Again, a consideration for migrating an older codebase.

// 1. If a writer sets Reference in v1, do they also set Type in v2?  One piece of information in v1 implies two fields in v2.

// If you say "yes" to case 1, then you're implicitly running a defaulter in conversion, you've just called it something different and moved it to the internal->v2 conversion, which never happened before.
// If you say "no" in case 1, then your conversion logic had to duplicate the defaulter logic to decide which Reference to set and then know to avoid persisting the result of that defaulting.
//   You're also not expressing the full intent of the user in the applied object serialization.
// I suspect you intend "no", but duplicating logic in converters is concerning.
// Making it a case-by-case will make future API reviews even harder.
// Having to logically handle defaulting on outbound conversions is net new.

type Foo struct {
	metav1.TypeMeta
	metav1.Object

	// BarReference refers to another name.  The kind of object is always Bar.
	BarReference string

	// BazReference refers to another name.  The kind of object is always Baz.
	BazReference string

	// Type says which to use (some of our older APIs did this), original VolumeSource as a for instance.  We never supported empty.
	Type string
}

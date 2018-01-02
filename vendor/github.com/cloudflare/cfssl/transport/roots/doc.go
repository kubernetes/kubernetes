// Package roots includes support for loading trusted roots from
// various sources.
//
// The following are supported trusted roout sources provided:
//
// The "system" type does not take any metadata. It will use the
// default system certificates provided by the operating system.
//
// The "cfssl" provider takes keys for the CFSSL "host", "label", and
// "profile", and loads the returned certificate into the trust store.
//
// The "file" provider takes a source file (specified under the
// "source" key) that contains one or more certificates and adds
// them into the source tree.
package roots

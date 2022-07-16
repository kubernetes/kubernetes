// Package cdi has the primary purpose of providing an API for
// interacting with CDI and consuming CDI devices.
//
// For more information about Container Device Interface, please refer to
// https://github.com/container-orchestrated-devices/container-device-interface
//
// Container Device Interface
//
// Container Device Interface, or CDI for short, provides comprehensive
// third party device support for container runtimes. CDI uses vendor
// provided specification files, CDI Specs for short, to describe how a
// container's runtime environment should be modified when one or more
// of the vendor-specific devices is injected into the container. Beyond
// describing the low level platform-specific details of how to gain
// basic access to a device, CDI Specs allow more fine-grained device
// initialization, and the automatic injection of any necessary vendor-
// or device-specific software that might be required for a container
// to use a device or take full advantage of it.
//
// In the CDI device model containers request access to a device using
// fully qualified device names, qualified names for short, consisting of
// a vendor identifier, a device class and a device name or identifier.
// These pieces of information together uniquely identify a device among
// all device vendors, classes and device instances.
//
// This package implements an API for easy consumption of CDI. The API
// implements discovery, loading and caching of CDI Specs and injection
// of CDI devices into containers. This is the most common functionality
// the vast majority of CDI consumers need. The API should be usable both
// by OCI runtime clients and runtime implementations.
//
// CDI Registry
//
// The primary interface to interact with CDI devices is the Registry. It
// is essentially a cache of all Specs and devices discovered in standard
// CDI directories on the host. The registry has two main functionality,
// injecting devices into an OCI Spec and refreshing the cache of CDI
// Specs and devices.
//
// Device Injection
//
// Using the Registry one can inject CDI devices into a container with code
// similar to the following snippet:
//
//  import (
//      "fmt"
//      "strings"
//
//      "github.com/pkg/errors"
//      log "github.com/sirupsen/logrus"
//
//      "github.com/container-orchestrated-devices/container-device-interface/pkg/cdi"
//      oci "github.com/opencontainers/runtime-spec/specs-go"
//  )
//
//  func injectCDIDevices(spec *oci.Spec, devices []string) error {
//      log.Debug("pristine OCI Spec: %s", dumpSpec(spec))
//
//      unresolved, err := cdi.GetRegistry().InjectDevices(spec, devices)
//      if err != nil {
//          return errors.Wrap(err, "CDI device injection failed")
//      }
//
//      log.Debug("CDI-updated OCI Spec: %s", dumpSpec(spec))
//      return nil
//  }
//
// Cache Refresh
//
// In a runtime implementation one typically wants to make sure the
// CDI Spec cache is up to date before performing device injection.
// A code snippet similar to the following accmplishes that:
//
//  import (
//      "fmt"
//      "strings"
//
//      "github.com/pkg/errors"
//      log "github.com/sirupsen/logrus"
//
//      "github.com/container-orchestrated-devices/container-device-interface/pkg/cdi"
//      oci "github.com/opencontainers/runtime-spec/specs-go"
//  )
//
//  func injectCDIDevices(spec *oci.Spec, devices []string) error {
//      registry := cdi.GetRegistry()
//
//      if err := registry.Refresh(); err != nil {
//          // Note:
//          //   It is up to the implementation to decide whether
//          //   to abort injection on errors. A failed Refresh()
//          //   does not necessarily render the registry unusable.
//          //   For instance, a parse error in a Spec file for
//          //   vendor A does not have any effect on devices of
//          //   vendor B...
//          log.Warnf("pre-injection Refresh() failed: %v", err)
//      }
//
//      log.Debug("pristine OCI Spec: %s", dumpSpec(spec))
//
//      unresolved, err := registry.InjectDevices(spec, devices)
//      if err != nil {
//          return errors.Wrap(err, "CDI device injection failed")
//      }
//
//      log.Debug("CDI-updated OCI Spec: %s", dumpSpec(spec))
//      return nil
//  }
//
// Generated Spec Files, Multiple Directories, Device Precedence
//
// There are systems where the set of available or usable CDI devices
// changes dynamically and this needs to be reflected in the CDI Specs.
// This is done by dynamically regenerating CDI Spec files which are
// affected by these changes.
//
// CDI can collect Spec files from multiple directories. Spec files are
// automatically assigned priorities according to which directory they
// were loaded from. The later a directory occurs in the list of CDI
// directories to scan, the higher priority Spec files loaded from that
// directory are assigned to. When two or more Spec files define the
// same device, conflict is resolved by chosing the definition from the
// Spec file with the highest priority.
//
// The default CDI directory configuration is chosen to encourage
// separating dynamically generated CDI Spec files from static ones.
// The default directories are '/etc/cdi' and '/var/run/cdi'. By putting
// dynamically generated Spec files under '/var/run/cdi', those take
// precedence over static ones in '/etc/cdi'.
//
// CDI Spec Validation
//
// This package performs both syntactic and semantic validation of CDI
// Spec file data when a Spec file is loaded via the registry or using
// the ReadSpec API function. As part of the semantic verification, the
// Spec file is verified against the CDI Spec JSON validation schema.
//
// If a valid externally provided JSON validation schema is found in
// the filesystem at /etc/cdi/schema/schema.json it is loaded and used
// as the default validation schema. If such a file is not found or
// fails to load, an embedded no-op schema is used.
//
// The used validation schema can also be changed programmatically using
// the SetSchema API convenience function. This function also accepts
// the special "builtin" (BuiltinSchemaName) and "none" (NoneSchemaName)
// schema names which switch the used schema to the in-repo validation
// schema embedded into the binary or the now default no-op schema
// correspondingly. Other names are interpreted as the path to the actual
/// validation schema to load and use.
package cdi

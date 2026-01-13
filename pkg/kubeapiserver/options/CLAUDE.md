# Package: options

## Purpose
This package provides command-line options and configuration for the kube-apiserver, including defaults for service networking, admission plugins, authentication, and authorization settings.

## Key Types

- **AdmissionOptions**: Kube-apiserver-specific admission configuration wrapping generic options

## Key Constants and Variables

- **DefaultServiceNodePortRange**: Default NodePort range (30000-32767)
- **DefaultServiceIPCIDR**: Default service IP range (10.0.0.0/24)
- **DefaultEtcdPathPrefix**: Default etcd key prefix (/registry)

## Key Functions

- **NewAdmissionOptions()**: Creates admission options with all registered plugins
- **AddFlags()**: Adds kube-apiserver specific flags to FlagSet
- **Validate()**: Validates admission plugin configuration
- **ApplyTo()**: Applies admission configuration to server config

## Admission Plugin Management

- Registers all kube-apiserver admission plugins
- Provides RecommendedPluginOrder for default admission chain
- Supports enable/disable flags and deprecated --admission-control flag
- Computes enabled/disabled plugins from explicit list

## Design Notes

- Wraps generic apiserver options with Kubernetes-specific defaults
- Admission plugins registered via RegisterAllAdmissionPlugins()
- Options for authentication, authorization, and serving defined in separate files
- Maintains backward compatibility with deprecated flags

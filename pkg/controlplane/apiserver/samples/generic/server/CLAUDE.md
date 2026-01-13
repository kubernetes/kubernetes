# Package: server

## Purpose
This package implements the server logic for the sample-generic-controlplane. It wires together the control plane, API extensions (CRDs), and aggregator layers into a complete API server.

## Key Types

- **Config / CompletedConfig**: Configuration structs combining options with configs for Aggregator, ControlPlane, and APIExtensions servers
- **ExtraConfig**: Placeholder for sample-specific extra configuration
- **clientGetter**: ServiceAccountTokenGetter implementation that works without pod/node APIs

## Key Functions

- **NewCommand()**: Creates a cobra.Command for the sample-generic-apiserver with all flags and run logic
- **NewOptions()**: Creates server options with sample-specific defaults (admission plugins, cert directory, token getter)
- **Run()**: Main run function that creates config, builds server chain, and starts serving
- **NewConfig()**: Creates all server configurations using the controlplane apiserver package
- **CreateServerChain()**: Builds the delegation chain: Aggregator -> ControlPlane -> APIExtensions -> NotFoundHandler
- **DefaultOffAdmissionPlugins()**: Returns admission plugins that should be off by default for this sample

## Key Admission Plugins (On by Default)

- NamespaceLifecycle, ServiceAccount, DefaultTolerationSeconds
- MutatingAdmissionWebhook, ValidatingAdmissionWebhook
- ResourceQuota, CertificateApproval, CertificateSigning
- ValidatingAdmissionPolicy, MutatingAdmissionPolicy

## Design Notes

- Server chain uses delegation pattern: requests flow through aggregator, then control plane, then CRD handler
- ServiceAccount authentication works without pods/nodes by using a custom genericTokenGetter
- Disables RemoteAvailableConditionController since this is a standalone sample
- Uses the standard Kubernetes API server framework with customized admission defaults

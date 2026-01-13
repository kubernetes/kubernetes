# Package: bootstrap

## Purpose
Implements controllers for managing bootstrap tokens used during cluster setup, including ConfigMap signing and token cleanup.

## Key Types/Structs
- `Signer`: Controller that signs a ConfigMap with bootstrap tokens using JWS (JSON Web Signature)
- `SignerOptions`: Configuration for the Signer including namespaces and resync periods
- `TokenCleaner`: Controller that removes expired bootstrap tokens

## Key Functions
- `NewSigner(client, secrets, configMaps, options)`: Creates a new ConfigMap signer controller
- `DefaultSignerOptions()`: Returns default options (public namespace, cluster-info ConfigMap)
- `NewTokenCleaner(client, secrets)`: Creates a new token cleanup controller

## Signer Behavior
- Watches the cluster-info ConfigMap in kube-public namespace
- Watches bootstrap token secrets in kube-system namespace
- Signs ConfigMap data with each valid bootstrap token
- Stores signatures as annotations on the ConfigMap

## TokenCleaner Behavior
- Periodically scans bootstrap token secrets
- Deletes tokens that have expired based on their expiration time
- Runs on a configurable polling interval

## Design Notes
- Bootstrap tokens enable secure cluster joining without pre-shared credentials
- JWS signatures allow nodes to verify ConfigMap authenticity
- Part of the kubeadm bootstrapping workflow

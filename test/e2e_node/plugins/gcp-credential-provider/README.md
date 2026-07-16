# GCP credential provider for e2e testing

This package contains a barebones implementation of the [kubelet GCP credential
provider](https://github.com/kubernetes/cloud-provider-gcp/tree/master/cmd/auth-provider-gcp)
for testing purposes only. This plugin SHOULD NOT be used in production.

This credential provider is installed and configured in the node e2e tests by:

1. Building the gcp-credential-provider binary and including it in the test archive
   uploaded to the GCE remote node.

2. Writing the credential provider config into the temporary workspace consumed
  by the kubelet. The contents of the config should be something like this:

```yaml
kind: CredentialProviderConfig
apiVersion: kubelet.config.k8s.io/v1alpha1
providers:
  - name: gcp-credential-provider
    apiVersion: credentialprovider.kubelet.k8s.io/v1alpha1
    matchImages:
    - "gcr.io"
    - "*.gcr.io"
    - "container.cloud.google.com"
    - "*.pkg.dev"
    defaultCacheDuration: 1m`
```

3. Configuring the following additional flags on the kubelet:

```
--feature-gates=DisableKubeletCloudCredentialProviders=true
--image-credential-provider-config=/tmp/node-e2e-123456/credential-provider.yaml
--image-credential-provider-bin-dir=/tmp/node-e2e-12345
```

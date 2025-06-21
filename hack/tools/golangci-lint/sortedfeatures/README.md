# SortedFeatures Linter

This linter checks if feature gates in Kubernetes code are sorted alphabetically in const and var blocks.

## Purpose

In Kubernetes, feature gates should be listed in alphabetical, case-sensitive (upper before any lower case character) order to reduce the risk of code conflicts and improve readability. This linter enforces this convention by checking if feature gates are properly sorted.

## Installation

### As a golangci-lint plugin

1. Build the plugin:

```bash
cd hack/tools/golangci-lint/sortedfeatures
go build -buildmode=plugin -o sortedfeatures.so ./plugin/example.go
```

2. Add the plugin to your `.golangci.yml` configuration:

```yaml
linters:
  settings:
    custom:
      sortedfeatures:
        path: /path/to/sortedfeatures.so
        description: Checks if feature gates are sorted alphabetically
        original-url: k8s.io/kubernetes/hack/tools/golangci-lint/sortedfeatures
        settings:
          debug: false
          files:
            - path/to/additional/file.go
```

## Configuration Options

The linter supports the following configuration options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| debug | bool | false | Enable debug logging |
| files | []string | [] | Files to check for feature gate sorting. If specified, only these files will be checked (default files will be ignored) |
| paths | []string | [] | Alternative way to specify files (for compatibility). Only used if `files` is not specified |

You can specify files to check in two ways:

```yaml
# Using files (preferred)
settings:
  files:
    - path/to/file.go
    - another/path/file.go

# Using paths (alternative syntax)
settings:
  paths:
    - path/to/file.go
    - another/path/file.go
```

Note: If `files` is specified, only those files will be checked and the default files will be ignored. If neither `files` nor `paths` is specified, the default set of Kubernetes feature gate files will be checked.

## Usage

The linter will check all const and var blocks in your code to ensure that feature gates are sorted alphabetically. If they are not sorted, it will report an error with the current order and the expected order.

### Enabling the linter

Custom linters are enabled by default, but abide by the same rules as other linters.

If the disable all option is specified either on command line or in `.golangci.yml` files `linters.disable-all: true`, custom linters will be disabled;
they can be re-enabled by adding them to the `linters.enable` list,
or providing the enabled option on the command line, `golangci-lint run -Esortedfeatures`.

## Example

```go
const (
    // These are properly sorted
    FeatureA featuregate.Feature = "FeatureA"
    FeatureB featuregate.Feature = "FeatureB"
    FeatureC featuregate.Feature = "FeatureC"
)

const (
    // These are NOT properly sorted and will trigger a linter error
    FeatureB featuregate.Feature = "FeatureB"
    FeatureA featuregate.Feature = "FeatureA"
    FeatureC featuregate.Feature = "FeatureC"
)
```

## Files Checked

By default, this linter checks the following files in the Kubernetes codebase:

- `staging/src/k8s.io/apiserver/pkg/features/kube_features.go`
- `pkg/features/kube_features.go`
- `staging/src/k8s.io/client-go/features/known_features.go`
- `staging/src/k8s.io/controller-manager/pkg/features/kube_features.go`
- `staging/src/k8s.io/apiextensions-apiserver/pkg/features/kube_features.go`
- `test/e2e/feature/feature.go`
- `test/e2e/environment/environment.go`

## Integration with CI

This linter is part of the Kubernetes CI pipeline and helps ensure that all feature gates are properly sorted across the codebase. It's recommended to run this linter locally before submitting pull requests that modify feature gates.

## Troubleshooting

If you encounter issues with the linter:

1. Enable debug mode in your configuration
2. Check that the plugin is correctly built and referenced in your golangci-lint configuration
3. Verify that the files you want to check are either in the default list or explicitly specified in your configuration

For more information on custom linters in golangci-lint, refer to the [official documentation](https://golangci-lint.run/contributing/new-linters/).

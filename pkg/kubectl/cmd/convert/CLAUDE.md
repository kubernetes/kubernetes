# Package: convert

## Purpose
This package implements the `kubectl convert` command, which converts Kubernetes config files between different API versions. It supports both YAML and JSON formats.

## Key Types

- **ConvertOptions**: Holds data required to perform the convert operation

## Key Functions

- **NewCmdConvert()**: Creates the cobra command for convert
- **Complete()**: Collects information from command line to prepare conversion
- **RunConvert()**: Implements the generic convert command
- **asVersionedObject()**: Converts infos into a single versioned object or list
- **asVersionedObjects()**: Converts a list of infos into versioned objects
- **tryConvert()**: Attempts conversion to provided versions in order

## Usage Examples

```bash
# Convert pod.yaml to latest version
kubectl convert -f pod.yaml

# Convert to specific version with JSON output
kubectl convert -f pod.yaml --local -o json

# Convert all files in directory
kubectl convert -f . | kubectl create -f -
```

## Design Notes

- Uses legacy scheme for conversion (imports all API group installs)
- Defaults to YAML output format
- Supports --local flag to avoid contacting API server
- Falls back to object's mapping version if target version unavailable
- Warns on unregistered objects but continues processing
- Extensions API group imported last for proper version priority

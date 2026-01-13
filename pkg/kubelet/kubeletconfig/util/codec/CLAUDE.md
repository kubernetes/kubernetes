# Package: codec

Provides encoding and decoding utilities for KubeletConfiguration.

## Key Functions

- **EncodeKubeletConfig(internal, targetVersion)**: Encodes an internal KubeletConfiguration to external YAML representation for the specified API version.
- **NewKubeletconfigYAMLEncoder(targetVersion)**: Creates a YAML encoder for the kubeletconfig API group.
- **DecodeKubeletConfiguration(ctx, codecs, data)**: Decodes serialized KubeletConfiguration to the internal type. Supports lenient decoding fallback for v1beta1 compatibility.
- **DecodeKubeletConfigurationIntoJSON(codecs, data)**: Decodes configuration and returns it as JSON bytes with GroupVersionKind.

## Design Notes

- Uses runtime.Encode/Decode for proper versioning and defaulting
- Strict decoding is preferred, with lenient fallback for v1beta1 backwards compatibility
- Logs warnings when lenient decoding is used due to strict decoding failures
- The UniversalDecoder handles defaulting and conversion to internal types

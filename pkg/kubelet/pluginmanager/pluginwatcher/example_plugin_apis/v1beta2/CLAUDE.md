# Package: v1beta2

Protobuf-generated gRPC API for example plugins (version v1beta2).

## Generated Files

- **api.pb.go**: Protocol buffer message types
- **api_grpc.pb.go**: gRPC service client and server interfaces

## Purpose

This package provides a versioned gRPC API for testing the plugin watcher and registration mechanism. It represents a newer API version (v1beta2) compared to v1beta1.

## Proto Source

Generated from: `pkg/kubelet/pluginmanager/pluginwatcher/example_plugin_apis/v1beta2/api.proto`

## Usage

Used alongside v1beta1 to test plugin version negotiation and API compatibility in the plugin registration workflow.

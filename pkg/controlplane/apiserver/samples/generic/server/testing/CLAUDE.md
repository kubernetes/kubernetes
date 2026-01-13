# Package: testing

## Purpose
This package provides test utilities for starting a sample-generic-controlplane server in integration tests. It handles server setup, etcd connection, health checks, and teardown.

## Key Types

- **TestServer**: Represents a running test server with ClientConfig, ServerOpts, TearDownFn, TmpDir, EtcdClient, and EtcdStoragePrefix
- **TestServerInstanceOptions**: Options for customizing test server behavior (e.g., SkipHealthzCheck)
- **TearDownFunc**: Function type for cleaning up test server resources

## Key Functions

- **StartTestServer()**: Starts an etcd-backed sample-generic-controlplane for testing. Configures the server with test defaults, waits for health and default namespace creation, returns TestServer with client config
- **StartTestServerOrDie()**: Wrapper that calls t.Fatal on error
- **NewDefaultTestServerOptions()**: Returns default test server options

## Test Server Features

- Creates temporary directory for certificates and data
- Configures localhost listener on random free port
- Sets up test service account signing key
- Waits for /healthz to return 200
- Waits for default namespace to be created
- Provides etcd client for direct storage inspection
- Handles graceful shutdown and cleanup

## Design Notes

- Uses ktesting.TB interface for test logging and lifecycle
- Configures high QPS/Burst (1000/10000) on client config for test throughput
- Handles both go test and bazel test environments for finding test fixtures
- Sets logsapi.ReapplyHandling to ignore unchanged to support multiple server instances

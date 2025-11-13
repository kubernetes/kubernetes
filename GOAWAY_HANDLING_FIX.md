# Fix for Issue #135285: GOAWAY Handling in APIServer

## Problem
The apiserver is not handling GOAWAY events from downstream services gracefully when aggregating,
particularly under high load scenarios.

## Solution
This fix implements proper GOAWAY handling in the aggregation layer:

1. **Add graceful shutdown handlers** for HTTP/2 GOAWAY frames
2. **Implement connection pooling** to reuse healthy connections
3. **Add exponential backoff** for failed connection attempts
4. **Improve error logging** to better debug connection issues

## Changes Made
- Modified pkg/util/net/util.go to handle GOAWAY frames
- Added connection health checks in pkg/aggregator/metrics.go
- Enhanced error handling in staging/src/k8s.io/apiserver/pkg/server/routes/logs.go

## Testing
Unit tests have been added to verify:
- GOAWAY frames are handled gracefully
- Connections are properly recycled
- Backoff strategies work correctly under load

Fixes issue #135285

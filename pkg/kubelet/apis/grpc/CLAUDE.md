# Package grpc

Package grpc provides rate limiting utilities for gRPC servers used by the kubelet.

## Key Types

- `Limiter`: Interface for rate limiting with a single `Allow() bool` method, compatible with golang.org/x/time/rate.Limiter

## Key Functions

- `LimiterUnaryServerInterceptor`: Returns a gRPC unary server interceptor that rejects requests when the rate limit is exceeded
- `WithRateLimiter`: Creates a gRPC ServerOption with rate limiting configured using QPS and burst token parameters

## Key Variables

- `ErrorLimitExceeded`: Pre-defined gRPC error (codes.ResourceExhausted) returned when rate limit is hit

## Design Notes

- Uses token bucket algorithm via golang.org/x/time/rate
- Designed to protect kubelet gRPC endpoints from excessive requests
- Interceptor pattern allows easy integration with existing gRPC servers

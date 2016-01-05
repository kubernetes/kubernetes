# Authentication

As outlined <a href="https://github.com/grpc/grpc/blob/master/doc/grpc-auth-support.md">here</a> gRPC supports a number of different mechanisms for asserting identity between an client and server. We'll present some code-samples here demonstrating how to provide TLS support encryption and identity assertions as well as passing OAuth2 tokens to services that support it.

# Enabling TLS on a gRPC client

```Go
conn, err := grpc.Dial(serverAddr, grpc.WithTransportCredentials(credentials.NewClientTLSFromCert(nil, ""))
```

# Enabling TLS on a gRPC server

```Go
creds, err := credentials.NewServerTLSFromFile(certFile, keyFile)
if err != nil {
  log.Fatalf("Failed to generate credentials %v", err)
}
lis, err := net.Listen("tcp", ":0")
server := grpc.NewServer(grpc.Creds(creds))
...
server.Serve(lis)
```

# Authenticating with Google

## Google Compute Engine (GCE)

```Go
conn, err := grpc.Dial(serverAddr, grpc.WithTransportCredentials(credentials.NewClientTLSFromCert(nil, ""), grpc.WithPerRPCCredentials(oauth.NewComputeEngine())))
```

## JWT

```Go
jwtCreds, err := oauth.NewServiceAccountFromFile(*serviceAccountKeyFile, *oauthScope)
if err != nil {
  log.Fatalf("Failed to create JWT credentials: %v", err)
}
conn, err := grpc.Dial(serverAddr, grpc.WithTransportCredentials(credentials.NewClientTLSFromCert(nil, ""), grpc.WithPerRPCCredentials(jwtCreds)))
```


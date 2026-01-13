# Package: routes

## Purpose
Provides optional HTTP handlers for the Kubernetes API server master. Contains handlers for serving logs and OpenID Connect metadata endpoints.

## Key Types

- **Logs**: HTTP handler for serving log files from /var/log.
- **OpenIDMetadataServer**: HTTP server for service account token issuer OIDC metadata.

## Key Functions

### Logs Handler
- **Logs.Install(c)**: Registers /logs endpoint on the restful container.
- **logFileHandler(req, resp)**: Serves individual log files from /var/log.
- **logFileListHandler(req, resp)**: Lists available log files.
- **logFileNameIsTooLong(filePath)**: Checks if filename exceeds 255 characters.

### OpenID Metadata Server
- **NewOpenIDMetadataServer(provider)**: Creates new OIDC metadata server.
- **OpenIDMetadataServer.Install(c)**: Registers /.well-known/openid-configuration and /openid/v1/jwks endpoints.
- **serveConfiguration(w, req)**: Serves OIDC discovery document.
- **serveKeys(w, req)**: Serves JSON Web Key Set (JWKS) for token verification.

## Constants

- **fileNameTooLong**: Platform-specific error for filename too long (ENAMETOOLONG on Unix).

## Design Notes

- Logs handler provides direct access to node logs for debugging.
- OIDC metadata enables service account token verification by external systems.
- Cache-Control headers are set for OIDC responses to enable caching.

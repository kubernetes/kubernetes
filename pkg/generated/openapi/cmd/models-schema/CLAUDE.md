# Package: models-schema

## Purpose
This is a command-line tool that outputs the OpenAPI v2 schema JSON containing all Kubernetes API type definitions. It reads the generated OpenAPI definitions and serializes them to stdout.

## Key Functions

- **main()**: Entry point that calls output() and handles errors
- **output()**: Builds and serializes the complete OpenAPI v2 Swagger document

## Output Format

- OpenAPI/Swagger 2.0 specification
- Contains definitions for all Kubernetes API types
- Info section with title "Kubernetes" and version "unversioned"

## Design Notes

- Reads definitions from GetOpenAPIDefinitions() in the openapi package
- Handles v2/v3 schema differences by checking ExtensionV2Schema extension
- If a type has an embedded v2 schema, uses that instead of the default
- Output is JSON written to stdout for piping to files or other tools
- Used for generating documentation and client SDKs

# Package: storage

## Purpose
The `storage` package provides a TableConvertor that wraps a TableGenerator to implement the REST storage table conversion interface.

## Key Types

- **TableConvertor**: Wraps a TableGenerator for use in REST storage.
  - Embeds `printers.TableGenerator`.

## Key Methods

- **ConvertToTable**: Converts runtime objects to metav1.Table.
  - Accepts context, object, and table options.
  - Handles metav1.TableOptions for NoHeaders setting.
  - Always uses Wide mode for complete output.

## Design Notes

- Bridge between printers package and REST storage layer.
- Used by API server to serve table format responses.
- Simple adapter pattern wrapping TableGenerator.
- Ensures consistent table generation across API endpoints.

# Package: printers

## Purpose
The `printers` package defines interfaces and utilities for printing Kubernetes resources in various formats, particularly human-readable tables.

## Key Interfaces

- **ResourcePrinter**: Prints runtime objects to a writer.
  - `PrintObj(obj, writer)`: Formats and prints an object.

- **TableGenerator**: Generates metav1.Table from runtime objects.
  - `GenerateTable(obj, options)`: Converts object to table format.

- **PrintHandler**: Registers print handlers for specific types.
  - `TableHandler(columns, printFunc)`: Registers a print function with column definitions.

## Key Types

- **ResourcePrinterFunc**: Function type implementing ResourcePrinter.
- **GenerateOptions**: Options for table generation (NoHeaders, Wide).
- **HumanReadableGenerator**: Implementation of TableGenerator using registered handlers.

## Key Functions

- **NewTableGenerator**: Creates a new HumanReadableGenerator.
- **ValidateRowPrintHandlerFunc**: Validates print handler function signature.

## Design Notes

- Uses reflection to register type-specific print handlers.
- Print handlers convert objects to []metav1.TableRow.
- Supports wide output with priority-based column visibility.
- Column definitions include metadata like format, type, description.
- Used by kubectl and API server for table output format.

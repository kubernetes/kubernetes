# Package: priorityclass

## Purpose
Implements the registry strategy for PriorityClass objects. PriorityClass is a cluster-scoped resource that defines scheduling priority for pods.

## Key Types

- **priorityClassStrategy**: Implements REST strategy for PriorityClass CRUD operations.

## Key Variables

- **Strategy**: Singleton instance for Create, Update, and Delete operations.

## Key Functions

- **NamespaceScoped()**: Returns false - PriorityClass is cluster-scoped.
- **PrepareForCreate(ctx, obj)**: No-op, PriorityClass has no status.
- **Validate(ctx, obj)**: Validates using validation.ValidatePriorityClass.
- **PrepareForUpdate(ctx, obj, old)**: No-op.
- **ValidateUpdate(ctx, obj, old)**: Validates updates.
- **AllowUnconditionalUpdate()**: Returns true.

## Design Notes

- Simple strategy with no status subresource.
- Used by the scheduler to determine pod priority during scheduling.

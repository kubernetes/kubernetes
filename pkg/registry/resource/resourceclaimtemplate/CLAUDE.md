# Package: resourceclaimtemplate

## Purpose
Implements the registry strategy for ResourceClaimTemplate objects in the DRA API. ResourceClaimTemplate is a namespaced resource that defines a template for creating ResourceClaims.

## Key Types

- **resourceClaimTemplateStrategy**: Implements REST strategy for ResourceClaimTemplate CRUD operations.

## Key Functions

- **NewStrategy(nsClient)**: Creates strategy with namespace client for admin access validation.
- **NamespaceScoped()**: Returns true - ResourceClaimTemplate is namespaced.
- **PrepareForCreate(ctx, obj)**: Sets Generation to 1.
- **Validate(ctx, obj)**: Validates the template and checks admin access authorization.
- **PrepareForUpdate(ctx, obj, old)**: Increments Generation on spec changes.
- **ValidateUpdate(ctx, obj, old)**: Validates updates and admin access.
- **Match(label, field)**: Returns a SelectionPredicate for filtering.
- **GetAttrs(obj)**: Returns labels and selectable fields.

## Design Notes

- Validates admin access for device requests using AuthorizedForAdmin.
- Templates serve as blueprints for ResourceClaim objects.

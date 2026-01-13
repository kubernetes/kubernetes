# Package: selectors

## Purpose
Provides BiMultimap, an efficient bi-directional mapping data structure for associating objects by labels and selectors. Used by controllers that need to track which selecting objects (e.g., Services) match which labeled objects (e.g., Pods).

## Key Types

- **BiMultimap**: Thread-safe bi-directional multimap supporting label-based associations.
- **Key**: Tuple of Name and Namespace identifying an object.
- **labeledObject**: Internal representation of objects with labels.
- **selectingObject**: Internal representation of objects with selectors.

## Key Functions

### Labeled Object Operations
- **Put(key, labels)**: Inserts or updates a labeled object and its associations.
- **Delete(key)**: Removes a labeled object and its associations.
- **Exists(key)**: Checks if a labeled object exists.
- **KeepOnly(keys)**: Retains only specified labeled objects, deleting others.

### Selecting Object Operations
- **PutSelector(key, selector)**: Inserts or updates a selecting object and computes associations.
- **DeleteSelector(key)**: Removes a selecting object and its associations.
- **SelectorExists(key)**: Checks if a selecting object exists.
- **KeepOnlySelectors(keys)**: Retains only specified selecting objects.

### Query Operations
- **Select(key)**: Returns all labeled objects matching a selecting object's selector.
- **ReverseSelect(key)**: Returns all selecting objects whose selectors match a labeled object.

### Utilities
- **Parse(s)**: Parses "namespace/name" string into a Key.
- **NewBiMultimap()**: Creates a new empty BiMultimap.

## Design Notes

- Caches selector-to-labels and labels-to-selector associations for O(1) lookups after initial computation.
- Uses reference counting for garbage collection of cached associations.
- All operations are protected by RWMutex for concurrent access.
- Associations are namespace-scoped (selectors only match objects in same namespace).

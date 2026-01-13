# Package: csimigration

## Purpose
Manages migration of in-tree volume plugins to CSI drivers, allowing seamless transition from legacy volume plugins to their CSI equivalents.

## Key Types/Structs
- `PluginManager` - Manages CSI migration state and plugin lookups

## Key Functions
- `NewPluginManager()` - Creates migration plugin manager
- `IsMigrationEnabledForPlugin()` - Checks if migration is enabled for a plugin
- `GetCSINameFromInTreeName()` - Maps in-tree plugin name to CSI driver name
- `GetInTreeNameFromCSIName()` - Maps CSI driver name to in-tree plugin name

## Design Patterns
- Feature gate controlled migration (per-plugin feature flags)
- Transparent translation between in-tree specs and CSI specs
- Supports gradual rollout with feature gates like CSIMigrationAWS, CSIMigrationGCE
- Uses csi-translation-lib for spec conversion

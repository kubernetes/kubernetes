# Package: profile

## Purpose
Manages scheduler profiles, which define different scheduling behaviors by configuring different sets of plugins and their weights.

## Key Types
- `Map` - Maps profile names to framework implementations
- `RecorderFactory` - Factory for creating event recorders per profile

## Key Functions
- `NewMap()` - Creates a map of scheduler profiles from configuration
- `NewRecorderFactory()` - Creates factory for profile-specific event recorders

## Design Patterns
- Supports multiple scheduling profiles in a single scheduler instance
- Each profile can have different plugin configurations
- Profiles are selected based on pod's `.spec.schedulerName`
- Uses factory pattern for creating profile-specific components

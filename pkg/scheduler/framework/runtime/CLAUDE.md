# Package: runtime

## Purpose
Implements the scheduler framework runtime, providing the core execution engine for scheduler plugins and the plugin lifecycle management.

## Key Types
- `frameworkImpl` - Main framework implementation that runs scheduler plugins
- `Registry` - Plugin registry mapping plugin names to factory functions
- `waitingPodsMap` - Thread-safe map tracking pods waiting for permits
- `frameworkOptions` - Configuration options for framework initialization

## Key Functions
- `NewFramework()` - Creates a new scheduler framework with registered plugins
- `RunFilterPlugins()` - Executes filter plugins to find feasible nodes
- `RunScorePlugins()` - Executes scoring plugins to rank feasible nodes
- `RunPreFilterPlugins()` / `RunPostFilterPlugins()` - Pre/post filter hooks
- `RunPermitPlugins()` - Handles permit decisions (allow/deny/wait)
- `RunBindPlugins()` - Executes binding plugins to assign pods to nodes

## Design Patterns
- Plugin-based architecture with well-defined extension points
- Parallel plugin execution with configurable parallelism
- Instrumented wrappers for metrics collection on plugin execution times
- Supports batching operations for efficiency
- Uses context-based cancellation for timeouts

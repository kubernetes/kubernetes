# Package: clusterroleaggregation

## Purpose
Implements a controller that combines ClusterRole rules from multiple ClusterRoles into aggregate ClusterRoles based on label selectors.

## Key Types/Structs
- `ClusterRoleAggregationController`: Controller with ClusterRole client, lister, and workqueue

## Key Functions
- `NewClusterRoleAggregation(clusterRoleInformer, clusterRoleClient)`: Creates the aggregation controller
- `Run(ctx, workers)`: Starts the controller
- `syncClusterRole(ctx, key)`: Synchronizes a single ClusterRole's aggregated rules

## Aggregation Logic
1. Find ClusterRoles with AggregationRule set
2. For each aggregation selector, find matching ClusterRoles by labels
3. Collect all rules from matched ClusterRoles
4. Sort and deduplicate rules
5. Apply aggregated rules to the target ClusterRole using server-side apply

## Behavior
- Watches all ClusterRole changes
- Requeues all aggregating roles on any ClusterRole change
- Uses server-side apply with "clusterrole-aggregation-controller" field manager
- Preserves existing fields managed by other controllers

## Use Case
Enables built-in roles like `admin`, `edit`, `view` to automatically include rules from addon-provided ClusterRoles that have matching aggregation labels.

## Design Notes
- Uses semantic equality for comparing PolicyRules
- Sorts rules for consistent ordering
- Any change to any ClusterRole triggers re-evaluation of all aggregators

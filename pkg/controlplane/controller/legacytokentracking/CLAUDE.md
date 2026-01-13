# Package: legacytokentracking

## Purpose
This controller maintains a ConfigMap that tracks when legacy service account token tracking was enabled in the cluster. The presence and timestamp in this ConfigMap indicates that all API servers in the cluster have the tracking feature enabled.

## Key Types

- **Controller**: Manages the tracking ConfigMap with rate-limited creation

## Key Constants

- ConfigMapName: "kube-apiserver-legacy-service-account-token-tracking"
- ConfigMapDataKey: "since"
- Namespace: kube-system
- Date format: "2006-01-02" (YYYY-MM-DD)

## Key Functions

- **NewController()**: Creates a controller with a 30-minute rate limiter for ConfigMap creation
- **Run() / RunWithContext()**: Starts the controller and syncs the ConfigMap
- **syncConfigMap()**: Creates the ConfigMap if missing, or fixes invalid date format

## Behavior

- Creates ConfigMap with current date when tracking is enabled
- Uses rate limiting (1 per 30 minutes) to prevent thrashing in mixed enabled/disabled HA clusters
- If ConfigMap exists with invalid date, updates it with current date
- ConfigMap existence signals to other components that tracking is active cluster-wide

## Design Notes

- Uses a filtered informer watching only the specific ConfigMap
- Rate limiting prevents constant create/delete cycles during rolling upgrades
- In HA clusters, ConfigMap will be created once all controllers enable the feature
- When feature is disabled, ConfigMap should be deleted (handled elsewhere)

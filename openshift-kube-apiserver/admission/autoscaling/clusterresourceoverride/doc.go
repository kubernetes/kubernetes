package clusterresourceoverride

// The ClusterResourceOverride plugin is only active when admission control config is supplied for it.
// The plugin allows administrators to override user-provided container request/limit values
// in order to control overcommit and optionally pin CPU to memory.
// The plugin's actions can be disabled per-project with the project annotation
// autoscaling.openshift.io/cluster-resource-override-enabled="false", so cluster admins
// can exempt infrastructure projects and such from the overrides.

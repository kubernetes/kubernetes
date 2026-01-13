# Package: validation

## Purpose
Provides validation functions for the Kubernetes scheduler configuration types. Ensures that scheduler configurations, profiles, plugins, and plugin arguments are valid before the scheduler starts.

## Key Functions

- **ValidateKubeSchedulerConfiguration(cc *config.KubeSchedulerConfiguration)**: Validates the entire scheduler configuration including client connection, leader election, parallelism, percentage of nodes to score, and all profiles.

- **ValidateKubeSchedulerProfile(path *field.Path, profile *config.KubeSchedulerProfile)**: Validates a single scheduler profile including its name, plugins configuration, and plugin weights.

- **ValidatePluginConfig(path *field.Path, profile *config.KubeSchedulerProfile)**: Validates plugin-specific arguments by dispatching to appropriate plugin argument validators.

## Plugin Argument Validators
- **ValidateDefaultPreemptionArgs**: Validates DefaultPreemption plugin arguments
- **ValidateInterPodAffinityArgs**: Validates InterPodAffinity plugin arguments
- **ValidateNodeAffinityArgs**: Validates NodeAffinity plugin arguments
- **ValidateNodeResourcesFitArgs**: Validates NodeResourcesFit plugin arguments
- **ValidatePodTopologySpreadArgs**: Validates PodTopologySpread plugin arguments
- **ValidateVolumeBindingArgs**: Validates VolumeBinding plugin arguments
- **ValidateDynamicResourcesArgs**: Validates DynamicResources plugin arguments

## Key Validations
- Profile names must be unique and valid DNS subdomain names
- Plugin weights must be within valid range (1-MaxTotalScore/MaxNodeScore)
- Parallelism must be positive
- Percentage of nodes to score must be between 0-100
- Required plugins (QueueSort, Bind) must be configured

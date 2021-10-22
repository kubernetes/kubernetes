# Change History

## Breaking Changes

### Removed Constants

1. ActionType.ActionTypeActionTypeAutomationRuleAction
1. ActionType.ActionTypeActionTypeModifyProperties
1. ActionType.ActionTypeActionTypeRunPlaybook
1. ConditionType.ConditionTypeConditionTypeAutomationRuleCondition
1. ConditionType.ConditionTypeConditionTypeProperty
1. Kind.KindKindAggregations
1. Kind.KindKindCasesAggregation

## Additive Changes

### New Constants

1. ActionType.ActionTypeAutomationRuleAction
1. ActionType.ActionTypeModifyProperties
1. ActionType.ActionTypeRunPlaybook
1. ConditionType.ConditionTypeAutomationRuleCondition
1. ConditionType.ConditionTypeProperty
1. Kind.KindAggregations
1. Kind.KindCasesAggregation

### New Funcs

1. CasesAggregationBySeverityProperties.MarshalJSON() ([]byte, error)
1. CasesAggregationByStatusProperties.MarshalJSON() ([]byte, error)
1. CloudErrorBody.MarshalJSON() ([]byte, error)
1. EntityAnalyticsProperties.MarshalJSON() ([]byte, error)
1. EyesOnSettingsProperties.MarshalJSON() ([]byte, error)
1. GeoLocation.MarshalJSON() ([]byte, error)
1. IPSyncerSettingsProperties.MarshalJSON() ([]byte, error)
1. IncidentAdditionalData.MarshalJSON() ([]byte, error)
1. Resource.MarshalJSON() ([]byte, error)
1. SecurityAlertPropertiesConfidenceReasonsItem.MarshalJSON() ([]byte, error)
1. ThreatIntelligence.MarshalJSON() ([]byte, error)

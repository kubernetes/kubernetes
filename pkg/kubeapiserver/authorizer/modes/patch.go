package modes

var ModeScope = "Scope"
var ModeSystemMasters = "SystemMasters"
var ModeMinimumKubeletVersion = "MinimumKubeletVersion"

func init() {
	AuthorizationModeChoices = append(AuthorizationModeChoices, ModeScope, ModeSystemMasters, ModeMinimumKubeletVersion)
}

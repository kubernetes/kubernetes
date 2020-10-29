package modes

var ModeScope = "Scope"
var ModeSystemMasters = "SystemMasters"

func init() {
	AuthorizationModeChoices = append(AuthorizationModeChoices, ModeScope, ModeSystemMasters)
}

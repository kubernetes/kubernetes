package gcp

func GetUpgradeTarget() string {
	return *upgradeTarget
}

func SetUpgradeTarget(val string) {
	upgradeTarget = &val
}

func GetUpgradeImage() string {
	return *upgradeImage
}

func SetUpgradeImage(val string) {
	upgradeImage = &val
}
